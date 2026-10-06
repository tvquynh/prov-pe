from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parents[1]
STUDY_A = Path(r'E:\phd\EVASIVE_MALWARE_RELIABILITY_THESIS\02_RQ1_RELIABILITY_UNDER_SHIFT\03_FINAL\STUDY_A_FINAL_CANDIDATE_V4R1_OFFICIAL_EMBER2024_20260922')
SOURCE = Path(r'E:\project_data\parquet_clean-week')
CACHE = Path(r'E:\phd\EVASIVE_MALWARE_RELIABILITY_THESIS\99_SCRATCH\study_a_v4r1_20260922\official_feature_cache')
SEEDS = [2026,2027,2028,2029,2030]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def log(event, **kw):
    print(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'event': event, **kw}), flush=True)
