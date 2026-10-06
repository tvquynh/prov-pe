"""Build a new collected-PE JSONL release from a completed VT snapshot, offline."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import ast
import gzip
import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
import time

BASE = Path(__file__).resolve().parents[1]
SOURCE = Path('E:/phase3/OPS_EMBER_FEATURES_20260915/out')
DB = BASE / 'snapshots/COLLECTED_PE_779619_20260923/COLLECTED_PE_779619_20260923.db'
PRIOR = Path('E:/phase3/OPS_EMBER_FEATURES_20260915/LABEL_FROM_HISTORICAL_VT_20260922/shard_manifest.json')
AUTHOR = Path('E:/ember2024/EMBER2024-main/EMBER2024-main/src/thrember')
SNAPSHOT = 'COLLECTED_PE_779619_20260923'
EXPECTED_DB = '2f8f86e94057cab6c561ce08ed0daf9cab5e4a71ec7ba703a3ca5e53d5508885'
EXPECTED_FEATURES = '58a085e9ad307aa2c52e165985ff80db8fd5b763891c0cba2d1758a4825f7273'
NULL_FIELDS = ['family', 'family_confidence', 'behavior', 'file_property', 'packer',
               'exploit', 'group', 'week_id', 'caps', 'ttps', 'mbc']


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def label(status, count):
    if status == 'not_found':
        return -1, 'VT_NOT_FOUND'
    if status != 'ok' or type(count) is not int or count < 0:
        raise ValueError('Invalid terminal status or malicious count')
    if count == 0:
        return 0, 'ZERO_MALICIOUS'
    if count >= 5:
        return 1, 'MALICIOUS_GE5'
    return -1, 'MALICIOUS_1_4'


def append_metadata(line, fields):
    body = line.rstrip(b'\r\n')
    if not body.endswith(b'}'):
        raise ValueError('Expected one JSON object per line')
    suffix = json.dumps(fields, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode()
    return body[:-1] + b',' + suffix[1:] + b'\n'


def test_contract():
    assert label('ok', 0) == (0, 'ZERO_MALICIOUS')
    assert label('ok', 1)[0] == -1 and label('ok', 4)[0] == -1
    assert label('ok', 5)[0] == 1 and label('ok', 90)[0] == 1
    assert label('not_found', None) == (-1, 'VT_NOT_FOUND')
    for status, count in [('failed_timeout', 0), ('ok', None), ('ok', -1), ('ok', True)]:
        try:
            label(status, count)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid source was accepted')
    original = b'{"sha256":"abc","strings":{"entropy":1.2345678901234567}}\r\n'
    combined = append_metadata(original, {'label': -1, 'caps': None})
    assert combined.startswith(original.rstrip(b'\r\n')[:-1])
    parsed = json.loads(combined)
    assert parsed['strings'] == json.loads(original)['strings'] and parsed['caps'] is None
    return 'PASS'


def author_reader():
    import numpy as np
    spec = importlib.util.spec_from_file_location('author_features', AUTHOR / 'features.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    extractor = mod.PEFeatureExtractor()
    # Run the exact author functions without importing the model-training module.
    tree = ast.parse((AUTHOR / 'model.py').read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('read_label', 'vectorize')]
    ns = {'json': json, 'np': np, 'PEFeatureExtractor': mod.PEFeatureExtractor}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(AUTHOR / 'model.py'), 'exec'), ns)
    return extractor, ns, np


def main(output):
    start = time.monotonic()
    assert test_contract() == 'PASS'
    if output.exists():
        raise FileExistsError('Refusing to overwrite output directory')
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Set PYTHONHASHSEED=0 for feature-reader verification')
    assert digest(AUTHOR / 'features.py') == EXPECTED_FEATURES
    assert digest(DB) == EXPECTED_DB
    assert not Path(str(DB) + '-wal').exists() or Path(str(DB) + '-wal').stat().st_size == 0
    prior = {r['file']: r for r in json.loads(PRIOR.read_text())}
    paths = sorted(SOURCE.glob('shard_*.jsonl'))
    assert [p.name for p in paths] == [f'shard_{i:04d}.jsonl' for i in range(256)]
    output.mkdir()
    audit = output / 'metadata'
    audit.mkdir()
    dump(audit / 'BUILD_STATE.json', {'status': 'BUILDING'})
    print('Source identity verified; reading terminal VT records', flush=True)
    c = sqlite3.connect(DB.as_uri() + '?mode=ro&immutable=1', uri=True)
    c.row_factory = sqlite3.Row
    cols = ['sha256', 'snapshot_id', 'terminal_request_id', 'terminal_status', 'malicious_count',
            'suspicious_count', 'stats_total', 'engine_roster_size', 'first_submission_date',
            'last_analysis_date', 'response_received_utc', 'original_label', 'file_type']
    by_sha = {}
    by_request = {}
    for r in c.execute('SELECT ' + ','.join(cols) + ' FROM snapshot_samples'):
        d = dict(r)
        assert d['snapshot_id'] == SNAPSHOT and d['sha256'] not in by_sha
        d['label'], d['reason'] = label(d['terminal_status'], d['malicious_count'])
        assert d['terminal_request_id'] not in by_request
        by_sha[d['sha256']] = d
        by_request[d['terminal_request_id']] = d
    assert len(by_sha) == 779619
    checked = set()
    for r in c.execute('SELECT request_id,sha256,snapshot_id,request_status,response_json_gzip FROM vt_observations'):
        d = by_request.get(r['request_id'])
        if d is None:
            continue
        assert r['sha256'] == d['sha256'] and r['snapshot_id'] == SNAPSHOT
        assert r['request_status'] == d['terminal_status']
        assert r['request_id'] not in checked
        checked.add(r['request_id'])
        if d['terminal_status'] == 'ok':
            assert r['response_json_gzip'] is not None
            report = json.loads(gzip.decompress(r['response_json_gzip']))
            attrs = report['data']['attributes']
            assert attrs['sha256'].lower() == d['sha256']
            stats = attrs['last_analysis_stats']
            assert stats['malicious'] == d['malicious_count']
            assert stats.get('suspicious', 0) == d['suspicious_count']
            assert sum(stats.values()) == d['stats_total']
            assert len(attrs.get('last_analysis_results', {})) == d['engine_roster_size']
            for key in ['first_submission_date', 'last_analysis_date']:
                assert attrs.get(key) == d[key]
            # Explicit denominator: all categories in VT last_analysis_stats.
            total = d['stats_total']
            d['detection_ratio'] = f"{d['malicious_count']}/{total}" if total > 0 else None
        else:
            d['detection_ratio'] = None
            assert d['first_submission_date'] is None and d['last_analysis_date'] is None
        if len(checked) % 100000 == 0:
            print('VT raw reports verified', len(checked), flush=True)
    c.close()
    assert len(checked) == len(by_sha)
    del checked, by_request
    extractor, reader, np = author_reader()
    import tempfile
    seen = set()
    manifest = []
    counts, reasons, filetypes = Counter(), Counter(), Counter()
    samples_tested = 0
    with tempfile.TemporaryDirectory(prefix='jsonl_reader_check_') as tmp:
        xp, yp = str(Path(tmp) / 'X.bin'), str(Path(tmp) / 'y.bin')
        for i, p in enumerate(paths):
            ih, oh, ph = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
            outp = output / p.name
            pp = audit / (p.stem + '.provenance.jsonl')
            n = 0
            local_reasons = set()
            with p.open('rb') as fin, outp.open('xb') as fout, pp.open('xb') as provenance:
                for line in fin:
                    ih.update(line)
                    raw = json.loads(line)
                    sha = raw['sha256']
                    assert sha not in seen and sha in by_sha and 'label' not in raw
                    seen.add(sha)
                    d = by_sha[sha]
                    assert raw['file_type'] == d['file_type']
                    fields = {'first_submission_date': d['first_submission_date'],
                              'last_analysis_date': d['last_analysis_date'],
                              'detection_ratio': d['detection_ratio'], 'label': d['label']}
                    fields.update({key: None for key in NULL_FIELDS})
                    assert not set(fields).intersection(raw)
                    result = append_metadata(line, fields)
                    fout.write(result)
                    oh.update(result)
                    pr = {'sha256': sha, 'corpus': raw['corpus'], 'source_shard': p.name,
                          'source_line': n + 1, 'snapshot_id': SNAPSHOT,
                          'terminal_request_id': d['terminal_request_id'],
                          'vt_status': d['terminal_status'], 'label': d['label'],
                          'label_reason': d['reason'], 'historical_label': d['original_label'],
                          'malicious_count': d['malicious_count'],
                          'suspicious_count': d['suspicious_count'],
                          'stats_total': d['stats_total'], 'engine_roster_size': d['engine_roster_size'],
                          'response_received_utc': d['response_received_utc']}
                    pb = (json.dumps(pr, separators=(',', ':'), allow_nan=False) + '\n').encode()
                    provenance.write(pb)
                    ph.update(pb)
                    counts[str(d['label'])] += 1
                    reasons[d['reason']] += 1
                    filetypes[raw['file_type']] += 1
                    n += 1
                    # One example per reason in every shard, deterministic selection.
                    if d['reason'] not in local_reasons:
                        local_reasons.add(d['reason'])
                        obj = json.loads(result)
                        assert reader['read_label'](result, 'label') == d['label']
                        before = extractor.process_raw_features(raw).astype(np.float32)
                        after = extractor.process_raw_features(obj).astype(np.float32)
                        assert before.shape == (2568,) and np.isfinite(after).all()
                        assert np.array_equal(before, after)
                        x = np.memmap(xp, dtype=np.float32, mode='w+', shape=(1, 2568)); del x
                        y = np.memmap(yp, dtype=np.int32, mode='w+', shape=(1,)); del y
                        reader['vectorize'](0, result, xp, yp, extractor, 1, 'label', {})
                        x = np.memmap(xp, dtype=np.float32, mode='r', shape=(1, 2568))
                        y = np.memmap(yp, dtype=np.int32, mode='r', shape=(1,))
                        assert np.array_equal(x[0], before) and int(y[0]) == d['label']
                        del x, y
                        samples_tested += 1
            assert ih.hexdigest() == prior[p.name]['input_sha256']
            assert n == prior[p.name]['records']
            # Full read-back checks: original fields byte-preserved; new fields from VT.
            with p.open('rb') as original, outp.open('rb') as written, pp.open('rb') as prov:
                import itertools
                for old, new, pline in itertools.zip_longest(original, written, prov):
                    assert old is not None and new is not None and pline is not None
                    parsed = json.loads(new)
                    d = by_sha[parsed['sha256']]
                    assert new.startswith(old.rstrip(b'\r\n')[:-1] + b',')
                    assert parsed['label'] == d['label']
                    for key in ['first_submission_date', 'last_analysis_date', 'detection_ratio']:
                        assert parsed[key] == d[key]
                    assert all(parsed[key] is None for key in NULL_FIELDS)
                    pr = json.loads(pline)
                    assert pr['sha256'] == parsed['sha256'] and pr['terminal_request_id'] == d['terminal_request_id']
                    assert pr['label_reason'] == d['reason'] and pr['label'] == d['label']
            assert digest(outp) == oh.hexdigest() and digest(pp) == ph.hexdigest()
            manifest.append({'file':p.name,'rows':n,'input_sha256':ih.hexdigest(),
                             'output_sha256':oh.hexdigest(),'output_bytes':outp.stat().st_size,
                             'provenance_file':str(pp.relative_to(output)).replace('\\','/'),
                             'provenance_sha256':ph.hexdigest()})
            if (i + 1) % 16 == 0:
                print('Shards written and read-back verified', i + 1, 'rows', len(seen), flush=True)
    assert seen == set(by_sha)
    assert counts == {'0':196191, '1':553322, '-1':30106}
    assert reasons == {'ZERO_MALICIOUS':196191,'MALICIOUS_GE5':553322,'MALICIOUS_1_4':6935,'VT_NOT_FOUND':23171}
    report = {'status':'PASS','completed_utc':datetime.now(timezone.utc).isoformat(),
              'rows':len(seen),'shards':len(manifest),'label_counts':dict(counts),
              'label_reason_counts':dict(reasons),'file_type_counts':dict(filetypes),
              'raw_terminal_reports_checked':779619,'all_original_record_bytes_preserved':True,
              'all_rows_read_back':True,'all_input_checksums_match_historical_manifest':True,
              'all_output_checksums_read_back':True,'author_reader_sample_count':samples_tested,
              'author_binary_reader':'PASS','author_sample_vectors_unchanged':True,
              'synthetic_contract_tests':'PASS','db_sha256':EXPECTED_DB,
              'features_py_sha256':digest(AUTHOR/'features.py'),
              'model_py_sha256':digest(AUTHOR/'model.py'),'script_sha256':digest(Path(__file__)),
              'input_directory':str(SOURCE),'source_db':str(DB),'snapshot_id':SNAPSHOT,
              'python':sys.version,'numpy':np.__version__,'elapsed_seconds':round(time.monotonic()-start,2),
              'no_new_vt_requests':True,'no_training':True,'no_new_pe_extraction':True,
              'limitations':['Count-threshold labels; full EMBER2024 labeling conditions not certified.',
                             'Unknown tags are null; binary label reader verified, not multilabel training.',
                             'No train/test split; supply explicit shard paths to author reader.',
                             'Upstream feature parity FAIL remains unchanged.']}
    dump(audit/'shard_manifest.json',manifest)
    dump(audit/'VALIDATION_REPORT.json',report)
    dump(audit/'BUILD_STATE.json',{'status':'COMPLETE','rows':len(seen)})
    lines=[]
    for f in sorted(output.rglob('*')):
        if f.is_file():
            lines.append(digest(f)+'  '+str(f.relative_to(output)).replace('\\','/'))
    (output/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=BASE/'jsonl_vt_20260927')
    args = parser.parse_args()
    main(args.output.resolve())
