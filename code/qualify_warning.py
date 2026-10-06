"""Static diagnostic only: never execute a corpus binary or contact VT."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = Path(r'E:\ember2024\EMBER2024-main\EMBER2024-main\src\thrember\features.py')
PYTHON = Path(r'E:\phase3\OPS_EMBER_FEATURES_20260915\venv\Scripts\python.exe')
SHA = '2d683fa70bf9574ad512dc35b4d07839369c183cc2135d87340a83ea38953729'
BINARY = Path(r'E:\project_data\PE\EMBER2024\train') / SHA
ZIP = Path(r'E:\phd\EVASIVE_MALWARE_RELIABILITY_THESIS\99_SCRATCH\study_a_v4r1_20260922\original_release\Win64_train.zip')


def child():
    import numpy as np
    import pefile
    spec = importlib.util.spec_from_file_location('author_features', EXTRACTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extractor = module.PEFeatureExtractor()
    data = BINARY.read_bytes()
    assert hashlib.sha256(data).hexdigest() == SHA
    pe = pefile.PE(data=data)
    warning_feature = extractor.features[-1]
    raw = extractor.raw_features(data)
    vector = extractor.process_raw_features(raw)
    matches = []
    for warning in sorted(set(pe.get_warnings())):
        matches.append({'warning': warning,
                        'suffix_matches': sorted(s for s in warning_feature.warning_suffixes if warning.endswith(s)),
                        'prefix_matches': sorted(p for p in warning_feature.warning_prefixes if warning.startswith(p))})
    print('DIAGNOSTIC_JSON=' + json.dumps({
        'hash_seed': os.environ['PYTHONHASHSEED'], 'python': sys.version,
        'numpy': np.__version__, 'pefile': pefile.__version__,
        'normalized_warnings': raw['pefilewarnings'],
        'vector': vector.tolist(), 'warning_matches': matches,
        'feature_dimensions': [{'name': f.name, 'dim': f.dim} for f in extractor.features]
    }))


def main():
    out = ROOT / 'qualification'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'warning_diagnostic.json').exists(), 'Preserve completed diagnostic'
    with zipfile.ZipFile(ZIP) as archive:
        entry = next(n for n in archive.namelist() if n.endswith('2023-10-22_2023-10-28_Win64_train.jsonl'))
        with archive.open(entry) as handle:
            reference = next(json.loads(line) for line in handle if SHA.encode() in line[:400])
    (out / 'warning_reference_record.json').write_text(json.dumps(reference, sort_keys=True), encoding='utf8')
    runs = []
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 0]:
        env = {**os.environ, 'PYTHONHASHSEED': str(seed), 'PYTHONIOENCODING': 'utf-8'}
        result = subprocess.run([str(PYTHON), '-X', 'utf8', str(Path(__file__)), '--child'], env=env, capture_output=True, text=True, encoding='utf8', check=True)
        record = json.loads(next(line.split('=', 1)[1] for line in result.stdout.splitlines() if line.startswith('DIAGNOSTIC_JSON=')))
        runs.append(record)
        print('hash_seed', seed, 'normalized_warnings', record['normalized_warnings'], flush=True)
    import numpy as np
    spec = importlib.util.spec_from_file_location('author_features', EXTRACTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module.PEFeatureExtractor().process_raw_features(reference)
    for record in runs:
        vector = np.asarray(record.pop('vector'), dtype=np.float32)
        record['vector_sha256'] = hashlib.sha256(vector.tobytes()).hexdigest()
        record['reference_difference_dimensions'] = np.flatnonzero(vector != expected).tolist()
        record['differences'] = [{'dimension': int(i), 'actual': float(vector[i]), 'reference': float(expected[i])} for i in np.flatnonzero(vector != expected)]
    document = {
        'diagnostic_sample_sha256': SHA, 'binary_bytes': BINARY.stat().st_size,
        'author_extractor_sha256': hashlib.sha256(EXTRACTOR.read_bytes()).hexdigest(),
        'warning_vocabulary_sha256': hashlib.sha256(EXTRACTOR.with_name('pefile_warnings.txt').read_bytes()).hexdigest(),
        'reference_archive': str(ZIP), 'reference_entry': entry,
        'reference_warnings': reference['pefilewarnings'], 'runs': runs,
        'same_hash_seed_repeated_identically': runs[0]['vector_sha256'] == runs[-1]['vector_sha256'],
        'observed_distinct_warning_representations': len({tuple(r['normalized_warnings']) for r in runs}),
        'all_observed_differences_confined_to_warning_group': all(all(i >= 2480 for i in r['reference_difference_dimensions']) for r in runs),
        'scope': 'One prespecified discrepancy case; hash seeds are extractor diagnostics, not training seeds. No corpus-wide mismatch-rate estimate.'
    }
    (out / 'warning_diagnostic.json').write_text(json.dumps(document, indent=2), encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    child() if args.child else main()
