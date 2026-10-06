"""Verify named historical source-neighbor pairs, without looking at scores."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import tlsh
from common import *


def main():
    report=ROOT/'qualification/historical_near_pair_audit.json'
    assert not report.exists()
    assert not (ROOT/'results/per_seed_results.csv').exists(), 'This extension must precede external metric inspection'
    source_path=Path(r'E:\project_data\PE\BENIGN_PREVT_MASTER.parquet')
    columns=['sha256','collection_snapshot','ember_neardup','ember_neardup_min_distance','ember_neardup_reference_sha256','internal_neardup']
    prior=pd.read_parquet(source_path,columns=columns)
    assert prior.sha256.is_unique
    target=pd.read_parquet(ROOT/'data/target_manifest.parquet',columns=['sha256','tlsh_canonical','corpus','label','primary'])
    joined=target.merge(prior,on='sha256',how='left',validate='one_to_one')
    source=pd.concat([pd.read_parquet(ROOT/f'data/source_{s}_manifest.parquet',columns=['sha256','tlsh_canonical']) for s in ['train','test','challenge']],ignore_index=True)
    conflicts=source.groupby('sha256').tlsh_canonical.nunique()
    assert conflicts.max()==1
    source=source.drop_duplicates('sha256').rename(columns={'sha256':'ember_neardup_reference_sha256','tlsh_canonical':'source_tlsh'})
    joined=joined.merge(source,on='ember_neardup_reference_sha256',how='left',validate='many_to_one')
    distances=[]
    for a,b in zip(joined.tlsh_canonical,joined.source_tlsh):
        distances.append(tlsh.diff(a,b) if isinstance(a,str) and isinstance(b,str) else np.nan)
    joined['verified_distance']=distances
    joined['verified_source_neighbor_le30']=joined.verified_distance.le(30)
    joined['verified_nonidentical_source_neighbor_le30']=joined.verified_distance.gt(0)&joined.verified_distance.le(30)
    joined['primary_without_verified_source_neighbor']=joined.primary & ~joined.verified_source_neighbor_le30
    joined.to_parquet(ROOT/'data/historical_near_pair_ledger.parquet',index=False)
    proposal={
        'recorded_utc':datetime.now(timezone.utc).isoformat(),
        'role':'Outcome-blind additional sensitivity; original primary specification and original cohort masks retained',
        'reason':'Historical benign master contains named source-neighbor references that were not exposed by the original compact review package',
        'rule':'Exclude from primary only records whose historical named neighbor is in current pinned official PE source membership and whose independently recomputed TLSH distance is <=30. No searches selected by scores.',
        'threshold_origin':'EMBER2024 author paper near-duplicate distance rule; candidate pairs are pre-existing historical metadata',
        'scope':'Positive verification of named pairs. Missing candidate pairs do not establish absence of related files. Candidate-pair coverage is asymmetric by collection source.',
        'training_unchanged':True,'all_five_seeds_unchanged':True,'no_target_performance_read_or_computed':True,
        'base_experiment_sha256':sha(ROOT/'config/experiment.json'),
        'historical_master_sha256':sha(source_path),'ledger_sha256':sha(ROOT/'data/historical_near_pair_ledger.parquet'),
        'historically_flagged':int(joined.ember_neardup.fillna(False).sum()),
        'reference_in_current_source':int(joined.source_tlsh.notna().sum()),
        'verified_le30_inventory':int(joined.verified_source_neighbor_le30.sum()),
        'verified_nonidentical_le30_inventory':int(joined.verified_nonidentical_source_neighbor_le30.sum()),
        'verified_le30_primary':int((joined.primary&joined.verified_source_neighbor_le30).sum()),
        'retained_primary_label_counts':{str(k):int(v) for k,v in joined.loc[joined.primary_without_verified_source_neighbor,'label'].value_counts().items()},
        'collection_snapshots':{str(k):int(v) for k,v in joined.loc[joined.corpus.eq('benign_reference_candidate'),'collection_snapshot'].value_counts(dropna=False).items()}}
    save(report,proposal)
    save(ROOT/'config/outcome_blind_near_pair_extension.json',proposal)
    log('historical_pair_audit_complete',**{k:v for k,v in proposal.items() if k.endswith('inventory') or k.endswith('primary') or k=='retained_primary_label_counts'})


if __name__=='__main__':
    main()
