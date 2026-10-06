"""Fixed-cohort descriptive metrics and all-five-seed summaries; no resampling."""
import os
os.environ['OMP_NUM_THREADS']='4'
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from common import *


def metrics(frame, scores, threshold):
    labels=frame.label.to_numpy()
    negative=labels==0
    positive=labels==1
    predictions=scores>=threshold
    n0,n1=int(negative.sum()),int(positive.sum())
    fp,tp=int(predictions[negative].sum()),int(predictions[positive].sum())
    return {'n_negative':n0,'n_positive':n1,'false_positive':fp,'true_positive':tp,
            'FPR':fp/n0 if n0 else None,'TPR':tp/n1 if n1 else None,
            'ROC_AUC':float(roc_auc_score(labels,scores)) if n0 and n1 else None,
            'AP':float(average_precision_score(labels,scores)) if n0 and n1 else None,
            'positive_fraction':n1/(n0+n1),'sparse':n0<100 or n1<100}


def groups(frame):
    yield 'all','all',np.ones(len(frame),dtype=bool)
    for column in ['file_type','is_dll','corpus']:
        for value in sorted(frame[column].dropna().unique()):
            yield column,str(value),frame[column].eq(value).to_numpy()
    for (kind,dll), part in frame.groupby(['file_type','is_dll']):
        yield 'file_type_by_is_dll',f'{kind};DLL={dll}',frame.index.isin(part.index)
    age=pd.cut(frame.report_age_days,[-np.inf,30,365,np.inf],labels=['0_to_30','over30_to365','over365'])
    for value in age.cat.categories:
        yield 'report_age',str(value),age.eq(value).to_numpy()
    if age.isna().any():
        yield 'report_age','unavailable',age.isna().to_numpy()


def main():
    output=ROOT/'results'
    output.mkdir(exist_ok=True)
    cfg=read(ROOT/'config/experiment.json')
    freeze=read(ROOT/'config/analysis_freeze.json')
    assert sha(Path(__file__))==freeze['analysis_code_sha256']
    assert sha(ROOT/'config/outcome_blind_near_pair_extension.json')==freeze['near_extension_sha256']
    assert sha(ROOT/'data/evaluation_manifest.parquet')==cfg['target_manifest_sha256']
    target=pd.read_parquet(ROOT/'data/evaluation_manifest.parquet')
    test=pd.read_parquet(ROOT/'data/evaluation_source_test.parquet')
    test=test.loc[test.unique_sha_representative].reset_index(drop=True)
    overall=[]
    subgroups=[]
    thresholds=[]
    prediction_identity={}
    cohorts={'external_primary':np.ones(len(target),bool),'external_temporal':target.temporal_sensitivity.to_numpy(),
             'external_tls_equal':target.identical_tlsh_sensitivity.to_numpy(),
             'external_temporal_tls_equal':target.temporal_and_tlsh_sensitivity.to_numpy()}
    extension=read(ROOT/'config/outcome_blind_near_pair_extension.json')
    assert sha(ROOT/'data/historical_near_pair_ledger.parquet')==extension['ledger_sha256']
    near=pd.read_parquet(ROOT/'data/historical_near_pair_ledger.parquet').set_index('sha256').reindex(target.sha256)
    assert near.primary_without_verified_source_neighbor.notna().all()
    cohorts['external_known_near_control']=near.primary_without_verified_source_neighbor.to_numpy(dtype=bool)
    for seed in SEEDS:
        receipt=read(ROOT/f'logs/seed_{seed}_complete.json')
        assert receipt['experiment_sha256']==sha(ROOT/'config/experiment.json')
        for variant in ['prefix2480','full2568']:
            scores={}
            for name in ['source_test','external','validation']:
                path=ROOT/f'predictions/{variant}/{name}_seed_{seed}.npy'
                digest=sha(path)
                assert digest==receipt['variants'][variant]['scores_sha256'][name]
                prediction_identity[str(path.relative_to(ROOT))]=digest
                scores[name]=np.load(path)
            assert len(scores['external'])==len(target) and len(scores['source_test'])==len(test)
            for alpha in cfg['source_fpr_targets']:
                threshold=receipt['variants'][variant]['thresholds'][str(alpha)]
                thresholds.append({'variant':variant,'seed':seed,'source_fpr_target':alpha,'threshold':threshold,
                                   'validation_fpr':receipt['variants'][variant]['validation_actual_fpr'][str(alpha)],
                                   'validation_negative_n':receipt['variants'][variant]['validation_negative_n']})
                base={'variant':variant,'seed':seed,'source_fpr_target':alpha,'threshold':threshold}
                overall.append({**base,'cohort':'source_test',**metrics(test,scores['source_test'],threshold)})
                for name,mask in cohorts.items():
                    overall.append({**base,'cohort':name,**metrics(target.loc[mask],scores['external'][mask],threshold)})
                for grouping,value,mask in groups(target):
                    if mask.any():
                        subgroups.append({**base,'grouping':grouping,'group':value,**metrics(target.loc[mask],scores['external'][mask],threshold)})
        log('metrics_completed',seed=seed)
    main=pd.DataFrame(overall)
    detail=pd.DataFrame(subgroups)
    main.to_csv(output/'per_seed_results.csv',index=False)
    detail.to_csv(output/'per_seed_subgroups.csv',index=False)
    pd.DataFrame(thresholds).to_csv(output/'thresholds.csv',index=False)
    metric_names=['TPR','FPR','ROC_AUC','AP']
    records=[]
    for keys,group in main.groupby(['variant','cohort','source_fpr_target'],sort=True):
        assert sorted(group.seed.tolist())==SEEDS
        row=dict(zip(['variant','cohort','source_fpr_target'],keys))
        row.update(n_seeds=5,n_negative=int(group.n_negative.iloc[0]),n_positive=int(group.n_positive.iloc[0]))
        for metric in metric_names:
            row[metric+'_mean']=float(group[metric].mean())
            row[metric+'_sd']=float(group[metric].std(ddof=1))
        records.append(row)
    summary=pd.DataFrame(records)
    summary.to_csv(output/'summary_results.csv',index=False)
    sg=[]
    for keys,group in detail.groupby(['variant','source_fpr_target','grouping','group'],sort=True):
        row=dict(zip(['variant','source_fpr_target','grouping','group'],keys))
        row.update(n_seeds=len(group),n_negative=int(group.n_negative.iloc[0]),n_positive=int(group.n_positive.iloc[0]),sparse=bool(group['sparse'].iloc[0]))
        for metric in metric_names:
            row[metric+'_mean']=float(group[metric].mean()) if group[metric].notna().any() else None
            row[metric+'_sd']=float(group[metric].std(ddof=1)) if group[metric].notna().sum()>1 else None
        sg.append(row)
    pd.DataFrame(sg).to_csv(output/'subgroup_summary.csv',index=False)
    paired=[]
    for variant in ['prefix2480','full2568']:
        for alpha in cfg['source_fpr_targets']:
            ref=main.query('variant==@variant and source_fpr_target==@alpha and cohort=="external_primary"').set_index('seed')
            for cohort in ['external_temporal','external_tls_equal','external_temporal_tls_equal','external_known_near_control','source_test']:
                comparison=main.query('variant==@variant and source_fpr_target==@alpha and cohort==@cohort').set_index('seed')
                for seed in SEEDS:
                    paired.append({'comparison':cohort+' minus external_primary','variant':variant,'source_fpr_target':alpha,'seed':seed,
                                   **{metric+'_difference_pp':float(100*(comparison.loc[seed,metric]-ref.loc[seed,metric])) for metric in metric_names}})
    for cohort in main.cohort.unique():
        for alpha in cfg['source_fpr_targets']:
            a=main.query('cohort==@cohort and source_fpr_target==@alpha and variant=="prefix2480"').set_index('seed')
            b=main.query('cohort==@cohort and source_fpr_target==@alpha and variant=="full2568"').set_index('seed')
            for seed in SEEDS:
                paired.append({'comparison':'full2568 minus prefix2480','variant':cohort,'source_fpr_target':alpha,'seed':seed,
                               **{metric+'_difference_pp':float(100*(b.loc[seed,metric]-a.loc[seed,metric])) for metric in metric_names}})
    pd.DataFrame(paired).to_csv(output/'paired_differences.csv',index=False)
    pd.DataFrame(paired).groupby(['comparison','variant','source_fpr_target']).agg({metric+'_difference_pp':['mean','std'] for metric in metric_names}).to_csv(output/'paired_difference_summary.csv')
    save(output/'analysis_receipt.json',{'status':'COMPLETE','n_seeds':5,'no_bootstrap':True,'no_seed_hypothesis_tests':True,
                                       'experiment_sha256':sha(ROOT/'config/experiment.json'),'analysis_code_sha256':sha(Path(__file__)),
                                       'analysis_freeze_sha256':sha(ROOT/'config/analysis_freeze.json'),
                                       'prediction_identities':prediction_identity,'result_identities':{p.name:sha(p) for p in output.glob('*.csv')}})
    log('all_analysis_complete',overall_rows=len(main),subgroup_rows=len(detail))


if __name__=='__main__':
    main()
