import time
import numpy as np,pandas as pd,tlsh
from revision_common import *
def main():
    r=pd.read_parquet(ROOT/'data/exact_source_relatedness.parquet')
    sources=pd.read_parquet(ROOT/'data/relatedness_reference_digest_index.parquet').tlsh_canonical.tolist()
    old=pd.read_parquet(ORIGINAL/'data/historical_near_pair_ledger.parquet').set_index('sha256').reindex(r.sha256)
    assert (~old.verified_source_neighbor_le30.to_numpy(dtype=bool)|r.has_source_neighbor_le30.to_numpy()).all()
    # SHA-lexical, score-independent selection from each reported outcome, no model data.
    checks=[];started=time.monotonic()
    for expected in [False,True]:
        for row in r.loc[r.has_source_neighbor_le30.eq(expected)].sort_values('sha256').head(8).itertuples():
            seen=0;found=False
            for h in sources:
                seen+=1
                if tlsh.diff(row.tlsh_canonical,h)<=30:found=True;break
            assert found==expected
            checks.append({'sha256':row.sha256,'reported_neighbor':expected,'independent_library_neighbor':found,'library_comparisons':seen})
            log('independent_TLSH_query_verified',completed=len(checks))
    save(ROOT/'logs/relatedness_independent_verification.json',{'status':'PASS','all_historical_known_neighbors_recovered':True,'queries':checks,'elapsed_seconds':time.monotonic()-started,'method':'Direct py-tlsh diff scan, no index or custom distance function','ledger_sha256':sha(ROOT/'data/exact_source_relatedness.parquet')})
if __name__=='__main__':main()
