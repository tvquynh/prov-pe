"""Additional fixtures for explicit NULL handling in the metadata audit."""
import sqlite3
import tempfile
from contextlib import closing
from common import *
from metadata_audit import run


def main():
    checks={}
    with tempfile.TemporaryDirectory(prefix='metadata_null_') as tmp:
        for field in ['none','engine_roster_size','stats_total','stats_json']:
            path=Path(tmp)/(field+'.db')
            values={'engine_roster_size':2,'stats_total':2,'stats_json':'{"malicious":0,"undetected":2}'}
            if field!='none':values[field]=None
            with closing(sqlite3.connect(path)) as conn,conn:
                conn.execute('CREATE TABLE snapshot_samples(snapshot_id TEXT, sha256 TEXT, terminal_status TEXT, engine_roster_size INTEGER, stats_total INTEGER, stats_json TEXT)')
                conn.execute('INSERT INTO snapshot_samples VALUES(?,?,?,?,?,?)',('S','a'*64,'ok',values['engine_roster_size'],values['stats_total'],values['stats_json']))
            try:
                run(path,'S',sha(path),1,Path(tmp)/(field+'.json'))
            except AssertionError as ex:
                checks[field]=field!='none' and ('NULL' in str(ex) or 'lacks statistics JSON' in str(ex))
            else:
                checks[field]=field=='none'
    assert all(checks.values()),checks
    save(ROOT/'qualification/metadata_null_checks.json',{'status':'PASS','checks':checks,
        'metadata_code_sha256':sha(ROOT/'code/metadata_audit.py'),'sql_sha256':sha(ROOT/'code/metadata_audit.sql'),
        'scope':'Synthetic valid row plus NULL roster, NULL total and NULL statistics JSON rejection. Original nine-check receipt retained.'})
    log('metadata_null_fixtures_pass',checks=len(checks))


if __name__=='__main__':main()
