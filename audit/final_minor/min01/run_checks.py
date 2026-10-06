"""Execute the two authorized small test suites and record exact provenance."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
CODE = ROOT / 'candidate/code'
BASELINE = ROOT / 'baseline_readonly/code'
FIXTURE_PYTHON = ROOT.parent / 'minor_submission_revision_20260928_01/.venv/Scripts/python.exe'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_ast(path, name):
    module = ast.parse(Path(path).read_text(encoding='utf-8'))
    return ast.dump(next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name), include_attributes=False)


def utc():
    return datetime.now(timezone.utc).isoformat()


receipt_path = OUTPUT / 'execution_receipt.json'
if receipt_path.exists():
    raise SystemExit('Fresh execution receipt required')

unchanged_functions = {
    name: function_ast(BASELINE / 'reproduce_review.py', name) == function_ast(CODE / 'reproduce_review.py', name)
    for name in ['load_dependencies', 'digest', 'main']
}
unchanged_support = {
    name: sha(BASELINE / name) == sha(CODE / name)
    for name in ['fixture_profiles.py', 'cas_revision_common.py', 'test_fixture_profiles.py']
}
if not all(unchanged_functions.values()) or not all(unchanged_support.values()):
    raise SystemExit('Scientific checker preservation failed before tests')

commands = [
    ('fixture_profiles', [str(FIXTURE_PYTHON), '-B', '-O', '-X', 'utf8', str(CODE / 'test_fixture_profiles.py'), '--output', str(OUTPUT / 'fixture_profiles.json')]),
    ('receipt_provenance', [sys.executable, '-B', '-O', '-X', 'utf8', str(CODE / 'test_receipt_provenance.py'), '--output', str(OUTPUT / 'receipt_provenance.json')]),
]
receipt = {
    'started_utc': utc(), 'scope': 'MIN-01 metadata fix and existing synthetic regression tests only',
    'driver_argv': sys.argv, 'driver_python': sys.executable,
    'unchanged_function_ast': unchanged_functions, 'unchanged_support_file_bytes': unchanged_support,
    'source_sha256': {
        str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
        for p in [BASELINE / 'reproduce_review.py', CODE / 'reproduce_review.py',
                  CODE / 'test_receipt_provenance.py', CODE / 'test_fixture_profiles.py',
                  CODE / 'fixture_profiles.py', CODE / 'cas_revision_common.py', Path(__file__)]
    },
    'commands': [],
    'study_model_inference_calls': 0, 'scientific_metric_recomputations': 0,
    'training_runs': 0, 'VT_requests': 0, 'PE_parses': 0, 'bootstrap_runs': 0, 'new_experiments': 0,
}
for name, argv in commands:
    started = utc()
    timer = time.perf_counter()
    run = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    stdout = OUTPUT / (name + '.stdout.log')
    stderr = OUTPUT / (name + '.stderr.log')
    stdout.write_text(run.stdout, encoding='utf-8')
    stderr.write_text(run.stderr, encoding='utf-8')
    receipt['commands'].append({
        'name': name, 'argv': argv, 'cwd': str(ROOT), 'started_utc': started,
        'finished_utc': utc(), 'elapsed_seconds': time.perf_counter() - timer,
        'returncode': run.returncode, 'stdout_file': stdout.name, 'stderr_file': stderr.name,
        'stdout_sha256': sha(stdout), 'stderr_sha256': sha(stderr),
        'result_file': name + '.json', 'result_sha256': sha(OUTPUT / (name + '.json')),
    })
    print(name + ': return code ' + str(run.returncode), flush=True)
receipt['finished_utc'] = utc()
receipt['status'] = 'PASS' if all(item['returncode'] == 0 for item in receipt['commands']) else 'FAIL'
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': receipt['status'], 'receipt': str(receipt_path)}, indent=2))
raise SystemExit(0 if receipt['status'] == 'PASS' else 1)
