"""Replay exact PEFormatWarnings normalization on archived strings, without PE.

Only the pinned class AST is compiled. The complete extractor module is never
imported. The sole input object exposes archived strings via get_warnings().
"""
from pathlib import Path
import argparse
import ast
import json
import os
import subprocess
import sys
from cas_revision_common import require, sha, read, save, fresh, environment

CORE_SHA = '58a085e9ad307aa2c52e165985ff80db8fd5b763891c0cba2d1758a4825f7273'
VOCAB_SHA = 'a23a9d0a7a938b19390a75fe0eb024dbc9bad7a134bb1511a2913f365a52e5fb'


def child(root):
    directory = root / 'audit/warning_replay'
    core = directory / 'upstream/features.py'
    vocabulary = directory / 'upstream/pefile_warnings.txt'
    require(sha(core) == CORE_SHA and sha(vocabulary) == VOCAB_SHA, 'WARNING_IMPLEMENTATION_IDENTITY')
    tree = ast.parse(core.read_text(encoding='utf-8-sig'))
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PEFormatWarnings']
    require(len(selected) == 1, 'EXPECTED_SINGLE_WARNING_CLASS')
    namespace = {'Path': Path, 'FeatureType': object}
    compiled = compile(ast.Module(body=selected, type_ignores=[]), str(core), 'exec')
    exec(compiled, namespace)
    fixture = read(directory / 'fixture.json')

    class ArchivedWarningInput:
        def get_warnings(self):
            return fixture['raw_warning_strings']

    extractor = namespace['PEFormatWarnings'](vocabulary)
    result = extractor.raw_features(None, ArchivedWarningInput())
    print(json.dumps({'hash_seed': os.environ['PYTHONHASHSEED'], 'normalized_warnings': result,
                      'raw_warning_count': len(fixture['raw_warning_strings']),
                      'python': sys.version, 'PE_imported': 'pefile' in sys.modules}))


def main(args):
    out = fresh(args.output, [args.root])
    report = {'status': 'STARTED', 'environment': environment(), 'scope': 'Exact normalization class on archived warning strings; no binary extraction or feature re-extraction',
              'training_runs': 0, 'PE_parses': 0, 'PE_executions': 0, 'VT_requests': 0, 'new_experiments': 0}
    try:
        fixture_path = args.root / 'audit/warning_replay/fixture.json'
        fixture = read(fixture_path)
        runs = []
        for expected in fixture['expected_runs']:
            seed = expected['hash_seed']
            env = {**os.environ, 'PYTHONHASHSEED': str(seed), 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1'}
            command = [sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--root', str(args.root), '--child']
            completed = subprocess.run(command, env=env, capture_output=True, text=True, encoding='utf-8')
            require(completed.returncode == 0, 'CHILD_FAILED: ' + completed.stderr)
            observed = json.loads(completed.stdout)
            observed['expected_normalized_warnings'] = expected['normalized_warnings']
            observed['matches_archived_normalization'] = observed['normalized_warnings'] == expected['normalized_warnings']
            require(not observed['PE_imported'], 'UNEXPECTED_PE_IMPORT')
            runs.append(observed)
        save(out / 'observed_runs.json', runs)
        report.update(runs=runs, fixture_sha256=sha(fixture_path), extractor_sha256=CORE_SHA, vocabulary_sha256=VOCAB_SHA,
                      class_executed='PEFormatWarnings.__init__ and raw_features only',
                      normalization_runs=len(runs), distinct_normalizations=len({tuple(x['normalized_warnings']) for x in runs}),
                      repeated_seed_identical=runs[0]['normalized_warnings'] == runs[-1]['normalized_warnings'],
                      parser_output_origin='Historical warning_diagnostic.json; not reacquired',
                      historical_full_vector_failure_preserved=True)
        require(all(x['matches_archived_normalization'] for x in runs), 'RUNTIME_HASH_ORDER_DIFFERS_FROM_ARCHIVED_NORMALIZATION')
        report['status'] = 'VERIFIED_IMPLEMENTATION_REPLAY'
    except Exception as exc:
        report.update(status='REPLAY_FAILED', error=repr(exc))
    save(out / 'receipt.json', report)
    print(json.dumps({k: v for k, v in report.items() if k in ['status', 'error', 'normalization_runs', 'distinct_normalizations', 'repeated_seed_identical']}, indent=2))
    return 0 if report['status'] == 'VERIFIED_IMPLEMENTATION_REPLAY' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args(); args.root = args.root.resolve()
    if args.child:
        child(args.root)
    else:
        require(args.output is not None, 'OUTPUT_REQUIRED')
        sys.exit(main(args))
