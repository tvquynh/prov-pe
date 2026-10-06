"""Read-only metadata audit with verified identity and measured denominators."""
import argparse
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from common import *


def run(database, snapshot, expected_sha256, expected_rows, output):
    database=Path(database).resolve(strict=True)
    wal=Path(str(database)+'-wal')
    assert not wal.exists() or wal.stat().st_size==0, 'Uncheckpointed WAL is not permitted'
    actual=sha(database)
    assert actual==expected_sha256, 'Database SHA-256 does not match expected identity'
    with closing(sqlite3.connect(database.as_uri()+'?mode=ro&immutable=1',uri=True)) as connection:
        connection.row_factory=sqlite3.Row
        quality=dict(connection.execute('SELECT COUNT(*) rows, COUNT(DISTINCT sha256) unique_sha256, SUM(sha256 IS NULL) null_sha256 FROM snapshot_samples WHERE snapshot_id=?',(snapshot,)).fetchone())
        assert quality['rows']==quality['unique_sha256']==expected_rows
        assert quality['null_sha256']==0
        invalid=connection.execute("SELECT COUNT(*) FROM snapshot_samples WHERE snapshot_id=? AND (LENGTH(sha256) <> 64 OR sha256 GLOB '*[^0-9a-f]*')",(snapshot,)).fetchone()[0]
        assert invalid==0
        query_path=ROOT/'code/metadata_audit.sql'
        sql=query_path.read_text(encoding='utf8').replace("'COLLECTED_PE_779619_20260923'",':snapshot_id')
        chunks=re.split(r'^-- QUERY ([a-z_]+)\s*$',sql,flags=re.MULTILINE)
        results={}
        for i in range(1,len(chunks),2):
            name, statement=chunks[i],chunks[i+1].strip()
            assert statement.startswith('SELECT ')
            results[name]=[dict(r) for r in connection.execute(statement,{'snapshot_id':snapshot})]
        summary=results['engine_roster_and_stats_summary'][0]
        assert summary['reports_with_null_roster']==0, 'A usable report has NULL roster size'
        assert summary['reports_with_null_stats_total']==0, 'A usable report has NULL statistics total'
        assert results['missing_stats_json'][0]['reports_with_missing_stats_json']==0, 'A usable report lacks statistics JSON'
    document={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS','database':str(database),'snapshot_id':snapshot,
              'expected_sha256':expected_sha256,'observed_sha256':actual,'identity_matches':True,'measured_identity_checks':quality,
              'denominator_terminal_ok':results['engine_roster_and_stats_summary'][0]['terminal_ok_reports'],
              'sql_sha256':sha(query_path),'code_sha256':sha(Path(__file__)),'mode':'read_only_immutable','results':results}
    save(output,document)
    log('metadata_audit_complete',rows=quality['rows'],terminal_ok=document['denominator_terminal_ok'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--database',type=Path,default=DATASET/'snapshots/COLLECTED_PE_779619_20260923/COLLECTED_PE_779619_20260923.db')
    parser.add_argument('--snapshot',default='COLLECTED_PE_779619_20260923')
    parser.add_argument('--expected-sha256',default='2f8f86e94057cab6c561ce08ed0daf9cab5e4a71ec7ba703a3ca5e53d5508885')
    parser.add_argument('--expected-rows',type=int,default=779619)
    parser.add_argument('--output',type=Path,default=ROOT/'qualification/metadata_audit_verified.json')
    args=parser.parse_args()
    run(args.database,args.snapshot,args.expected_sha256,args.expected_rows,args.output)
