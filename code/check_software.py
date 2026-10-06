"""Synthetic guards and arithmetic checks; does not score real data."""
import ast
import sqlite3
import tempfile
from contextlib import closing
import numpy as np
import pandas as pd
from common import *
from run_experiment import threshold
from analyze_results import metrics
from metadata_audit import run as audit


def main():
    for path in (ROOT/'code').glob('*.py'):
        ast.parse(path.read_text(encoding='utf8'))
    checks={}
    t=threshold(np.array([.9,.9,.8,.1]),.25)
    checks['ties_not_split']=t>.9
    checks['zero_FP_above_max']=threshold(np.array([.2,.5]),0)>.5
    d=metrics(pd.DataFrame({'label':[0,0,1,1]}),np.array([.9,.1,.8,.7]),.5)
    checks['hand_counts']=d['true_positive']==2 and d['false_positive']==1 and d['TPR']==1 and d['FPR']==.5
    checks['hand_AUC']=d['ROC_AUC']==.5
    checks['hand_AP']=abs(d['AP']-(.5/2+(2/3)/2))<1e-12
    missing=metrics(pd.DataFrame({'label':[1,1]}),np.array([.9,.2]),.5)
    checks['single_class_undefined']=missing['ROC_AUC'] is None and missing['FPR'] is None and missing['AP'] is None
    with tempfile.TemporaryDirectory(prefix='collected_pe_software_') as tmp:
        assert Path(tmp).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
        path=Path(tmp)/'fixture.db'
        with closing(sqlite3.connect(path)) as c, c:
            c.execute('CREATE TABLE snapshot_samples(snapshot_id TEXT, sha256 TEXT, terminal_status TEXT, engine_roster_size INTEGER, stats_total INTEGER, stats_json TEXT)')
            c.executemany('INSERT INTO snapshot_samples VALUES(?,?,?,?,?,?)',[('FIXTURE','a'*64,'ok',2,2,'{"malicious":0,"undetected":2}'),('FIXTURE','b'*64,'not_found',0,0,None)])
        audit(path,'FIXTURE',sha(path),2,Path(tmp)/'valid.json')
        result=read(Path(tmp)/'valid.json')
        checks['fixture_measured_denominator']=result['denominator_terminal_ok']==1 and result['snapshot_id']=='FIXTURE'
        try:
            audit(path,'FIXTURE','0'*64,2,Path(tmp)/'invalid.json')
        except AssertionError:
            checks['wrong_database_rejected']=True
        else:
            checks['wrong_database_rejected']=False
        with closing(sqlite3.connect(path)) as c, c:
            c.execute('INSERT INTO snapshot_samples VALUES(?,?,?,?,?,?)',('FIXTURE','a'*64,'ok',2,2,'{"malicious":0,"undetected":2}'))
        try:
            audit(path,'FIXTURE',sha(path),3,Path(tmp)/'duplicate.json')
        except AssertionError:
            checks['duplicate_SHA_rejected']=True
        else:
            checks['duplicate_SHA_rejected']=False
    assert all(checks.values()),checks
    save(ROOT/'qualification/software_checks.json',{'status':'PASS','scope':'Synthetic fixtures only; no training or real prediction','checks':checks,'code_sha256':sha(Path(__file__))})
    log('software_checks_pass',checks=len(checks))


if __name__=='__main__':
    main()
