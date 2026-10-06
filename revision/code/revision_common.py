from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parents[1]
PROJECT=WORK.parent
ORIGINAL=PROJECT/'completion_20260928'
SEEDS=[2026,2027,2028,2029,2030]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def log(event,**kw):print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'event':event,**kw}),flush=True)
def threshold(scores,alpha):
    import numpy as np
    a,c=np.unique(np.asarray(scores,dtype=np.float64),return_counts=True)
    allowed=int(np.floor(alpha*len(scores)))
    valid=np.flatnonzero(np.cumsum(c[::-1])[::-1]<=allowed)
    return float(a[valid[0]]) if len(valid) else float(np.nextafter(a[-1],np.inf))

