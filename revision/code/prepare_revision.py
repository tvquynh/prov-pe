import platform,importlib.metadata
from revision_common import *
import numpy as np,pandas as pd

def main():
    out=ROOT/'config/revision_analysis.json'
    assert not out.exists()
    cache=Path(r'E:\phd\EVASIVE_MALWARE_RELIABILITY_THESIS\99_SCRATCH\study_a_v4r1_20260922\official_feature_cache')
    study=Path(r'E:\phd\EVASIVE_MALWARE_RELIABILITY_THESIS\02_RQ1_RELIABILITY_UNDER_SHIFT\03_FINAL\STUDY_A_FINAL_CANDIDATE_V4R1_OFFICIAL_EMBER2024_20260922')
    paths={'target_features':ORIGINAL/'cache/target.npy','source_train_features':cache/'train.npy','source_test_features':cache/'test.npy','target_metadata':ORIGINAL/'data/evaluation_manifest.parquet','source_train_metadata':ORIGINAL/'data/evaluation_source_train.parquet','source_test_metadata':ORIGINAL/'data/evaluation_source_test.parquet'}
    paths.update({f'split_{s}':study/f'data/internal_split_{s}.npz' for s in SEEDS})
    identity={}
    for k,p in paths.items():
        identity[k]={'path':str(p),'size':p.stat().st_size,'sha256':sha(p)};log('input_verified',name=k,sha256=identity[k]['sha256'])
    old=read(ORIGINAL/'config/experiment.json')
    assert identity['target_features']['sha256']==old['target_cache_sha256']
    assert identity['target_metadata']['sha256']==old['target_manifest_sha256']
    d=pd.read_parquet(paths['target_metadata'])
    assert len(d)==742273 and d.sha256.is_unique and d.primary.all()
    # No predictions or performance files are read in this preparation step.
    size=pd.cut(d.size_bytes,[-np.inf,65536,262144,1048576,4194304,np.inf],right=False,labels=['lt64KiB','64_256KiB','256KiB_1MiB','1_4MiB','ge4MiB']).astype(str)
    cell=d.file_type.astype(str)+'|'+d.is_dll.astype(str)+'|'+size
    counts=d.assign(cell=cell).groupby(['label','cell','corpus']).size().unstack(fill_value=0)
    supported=counts.index[counts.min(axis=1)>=20]
    membership=pd.MultiIndex.from_arrays([d.label,cell]).isin(supported)
    strata=d[['sha256','label','corpus','file_type','is_dll','size_bytes','report_age_days','malicious_count','tlsh_canonical']].copy()
    strata['cell']=cell;strata['common_support']=membership
    strata.to_parquet(ROOT/'data/revision_target_metadata.parquet',index=False)
    counts.to_csv(ROOT/'data/common_support_cell_counts.csv')
    splits={}
    for seed in SEEDS:
        bucket=d.tlsh_canonical.map(lambda v:int.from_bytes(hashlib.sha256(f'{seed}|{v}'.encode()).digest()[:8],'big')%1000).to_numpy()
        # Whole identical-TLSH groups are confined to one split, including mixed-label groups.
        split=np.where(bucket<600,0,np.where(bucket<800,1,2)).astype(np.int8)
        np.save(ROOT/f'data/source_diagnostic_split_{seed}.npy',split)
        splits[str(seed)]=d.assign(split=split).groupby(['split','corpus','label']).size().to_dict()
        assert set(split)=={0,1,2}
    cfg={'recorded_utc':datetime.now(timezone.utc).isoformat(),'role':'POST_REVIEW_SUPPLEMENTARY_ANALYSES','prior_primary_outcomes_known':True,'new_outcomes_not_read_at_specification':True,'seeds':SEEDS,'cpu_only':True,'threads':32,'max_new_fits':10,'inputs':identity,'feature_prefix':2480,'categorical_indices':[2,3,4,5,6,701,702],
      'source_diagnostic':{'learner':'LightGBM','positive_source':'mb_malware_candidate','group_split':'SHA256(seed|canonical_TLSH) modulo 1000; <600 train, 600-799 validation, >=800 test','parameters':'original fixed 500-round binary configuration, CPU32, experimental seeds substituted','decision_threshold':0.5,'no_early_stopping':True,'report':['overall','within_VT_label_0','within_VT_label_1'],'interpretation':'Source separability only; within-label test metrics do not turn pooled-source training into causal identification'},
      'common_support':{'strata':['VT_label','file_type','is_dll','size_bin'],'size_edges_bytes':[65536,262144,1048576,4194304],'min_per_source_per_label_cell':20,'method':'within each label, weight each source equally and each cell proportional to min(source counts); fixed saved malware scores and thresholds','preserve_natural_primary':True},
      'label_sensitivity':{'positive_minima':[5,10,15],'count_bins':['5-9','10-14','15-29','30+'],'negative_rule':'malicious_count == 0 unchanged','report_age_edges_days':[30,365],'no_ratio_relabeling':'engine independence/coverage not established','no_retraining':True},
      'linear_reference':{'estimator':'sklearn.linear_model.SGDClassifier','loss':'log_loss','penalty':'l2','alpha':0.0001,'max_iter':50,'tol':0.0001,'n_iter_no_change':5,'shuffle':True,'average':True,'learning_rate':'optimal','early_stopping':False,'preprocessing':'signed log1p then StandardScaler on 2473 noncategorical columns; OneHotEncoder(handle_unknown=ignore) on 7 categorical columns; fit only source fitting partition','split':'same saved EMBER2024 90/10 source splits as primary','thresholds':[0.01,0.001],'no_HPO':True,'nonconvergence':'report budget exhaustion honestly; no automatic alternative fit'},
      'summaries':'all five seeds, mean and sample SD ddof=1, no p-values or bootstrap','public_release':'DEFERRED_BY_AUTHOR_NOT_BLOCKING',
      'environment':{'python':platform.python_version(),'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','pyarrow','scikit-learn','lightgbm','scipy','py-tlsh']}}}
    cfg['prepared_data_hashes']={p.name:sha(p) for p in (ROOT/'data').iterdir() if p.is_file()}
    save(out,cfg)
    save(ROOT/'logs/preparation.json',{'status':'PASS','config_sha256':sha(out),'common_support_rows':int(membership.sum()),'split_counts':{s:{str(k):int(v) for k,v in c.items()} for s,c in splits.items()}})
    log('revision_specification_recorded',config_sha256=sha(out),common_support_rows=int(membership.sum()))
if __name__=='__main__':main()
