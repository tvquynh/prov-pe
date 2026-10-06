"""Supplementary fixed-score analyses; original results remain read-only."""
import os
os.environ['OMP_NUM_THREADS']='4'
import argparse
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from revision_common import *

def metric(y,s,t,w=None):
    y=np.asarray(y);s=np.asarray(s);w=np.ones(len(y)) if w is None else np.asarray(w)
    assert len(y)==len(s)==len(w) and np.isfinite(s).all() and (w>=0).all()
    z=y==0;o=y==1;pred=s>=t;both=z.any() and o.any()
    return {'n_negative':int(z.sum()),'n_positive':int(o.sum()),'FP':int(pred[z].sum()),'TP':int(pred[o].sum()),'FPR':float(np.average(pred[z],weights=w[z])) if z.any() and w[z].sum()>0 else None,'TPR':float(np.average(pred[o],weights=w[o])) if o.any() and w[o].sum()>0 else None,'ROC_AUC':float(roc_auc_score(y,s,sample_weight=w)) if both else None,'AP':float(average_precision_score(y,s,sample_weight=w)) if both else None}

def summarize(df,keys,out,metrics=('TPR','FPR','ROC_AUC','AP')):
    rows=[]
    for k,g in df.groupby(keys,dropna=False,sort=True):
        if not isinstance(k,tuple):k=(k,)
        row=dict(zip(keys,k));row['n_seeds']=len(g)
        for n in ['n_negative','n_positive','n_source0','n_source1']:
            if n in g:
                row[n+'_min']=int(g[n].min());row[n+'_max']=int(g[n].max())
        for m in metrics:
            row[m+'_mean']=float(g[m].mean()) if g[m].notna().any() else None
            row[m+'_sd']=float(g[m].std(ddof=1)) if g[m].notna().sum()>1 else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(out,index=False)

def metadata():
    cfg=read(ROOT/'config/revision_analysis.json')
    path=Path(cfg['inputs']['target_metadata']['path'])
    assert sha(path)==cfg['inputs']['target_metadata']['sha256']
    d=pd.read_parquet(path)
    a=pd.read_parquet(ROOT/'data/revision_target_metadata.parquet')
    assert np.array_equal(a.sha256,d.sha256)
    d['cell']=a.cell;d['common_support']=a.common_support
    return cfg,d

def existing():
    cfg,d=metadata();spec=cfg['label_sensitivity'];rows=[];conditionals=[];support=[]
    cells=pd.read_csv(ROOT/'data/common_support_cell_counts.csv').set_index(['label','cell'])
    sources=sorted(d.corpus.unique());mins=cells[sources].min(axis=1)
    weight=np.zeros(len(d))
    for label in [0,1]:
        good=(mins>=20)&(mins.index.get_level_values('label')==label);total=float(mins[good].sum())
        for cell,group in d.loc[d.common_support & d.label.eq(label)].groupby('cell'):
            mass=float(mins.loc[(label,cell)])/total
            for source in sources:
                ix=group.index[group.corpus.eq(source)];weight[ix]=mass/(2*len(ix))
    # Preserve the retained subset's class proportions when computing weighted AUC/AP.
    for label in [0,1]:weight[d.label.eq(label)]*=int((d.common_support & d.label.eq(label)).sum())
    np.save(ROOT/'data/common_support_weights.npy',weight)
    d.groupby(['corpus','label']).size().rename('n').to_csv(ROOT/'results/source_label_counts.csv')
    d.loc[d.common_support].groupby(['corpus','label']).size().rename('n').to_csv(ROOT/'results/common_support_counts.csv')
    ages=pd.cut(d.report_age_days,[-np.inf,30,365,np.inf],labels=['0-30','31-365','over365'])
    # Exact cohort denominator for the source-only descriptive rule.
    yy=d.label.to_numpy();source_rule=d.corpus.eq('mb_malware_candidate').to_numpy()
    save(ROOT/'results/source_rule_primary.json',{'n':len(d),'accuracy':float((source_rule==yy).mean()),'TPR':float(source_rule[yy==1].mean()),'TNR':float((~source_rule[yy==0]).mean()),'interpretation':'Describes collection-source/class association; not a trained malware detector'})
    for seed in SEEDS:
        old=read(ORIGINAL/f'logs/seed_{seed}_complete.json')['variants']['prefix2480']
        path=ROOT.parent/f'predictions/prefix2480/external_seed_{seed}.npy'
        assert sha(path)==old['scores_sha256']['external'];s=np.load(path)
        for alpha in [.01,.001]:
            t=old['thresholds'][str(alpha)];base={'seed':seed,'source_fpr_target':alpha,'threshold':t}
            for k in spec['positive_minima']:
                keep=(yy==0)|d.malicious_count.ge(k).to_numpy()
                rows.append({**base,'positive_minimum':k,**metric(yy[keep],s[keep],t)})
                for age in ['0-30','31-365','over365']:
                    select=keep&ages.eq(age).to_numpy()
                    conditionals.append({**base,'positive_minimum':k,'report_age':age,**metric(yy[select],s[select],t)})
            for name,mask,w in [('primary',np.ones(len(d),bool),None),('restricted_unweighted',d.common_support.to_numpy(),None),('source_standardized',d.common_support.to_numpy(),weight[d.common_support])]:
                support.append({**base,'view':name,'source':'pooled',**metric(yy[mask],s[mask],t,w)})
            for source in sources:
                mask=(d.common_support & d.corpus.eq(source)).to_numpy()
                support.append({**base,'view':'source_standardized','source':source,**metric(yy[mask],s[mask],t,weight[mask])})
    results=pd.DataFrame(rows);results.to_csv(ROOT/'results/label_sensitivity_per_seed.csv',index=False)
    summarize(results,['positive_minimum','source_fpr_target'],ROOT/'results/label_sensitivity_summary.csv')
    pd.DataFrame(conditionals).to_csv(ROOT/'results/label_by_age_per_seed.csv',index=False)
    summarize(pd.DataFrame(conditionals),['positive_minimum','report_age','source_fpr_target'],ROOT/'results/label_by_age_summary.csv')
    pd.DataFrame(support).to_csv(ROOT/'results/common_support_per_seed.csv',index=False)
    summarize(pd.DataFrame(support),['view','source','source_fpr_target'],ROOT/'results/common_support_summary.csv')
    # Counts and TPR within disjoint positive-count bands are also retained.
    bands=pd.cut(d.malicious_count,[4,9,14,29,np.inf],labels=spec['count_bins']);band_rows=[]
    for seed in SEEDS:
        s=np.load(ROOT.parent/f'predictions/prefix2480/external_seed_{seed}.npy');old=read(ORIGINAL/f'logs/seed_{seed}_complete.json')['variants']['prefix2480']
        for alpha in [.01,.001]:
            t=old['thresholds'][str(alpha)]
            for band in spec['count_bins']:
                mask=bands.eq(band).to_numpy();band_rows.append({'seed':seed,'source_fpr_target':alpha,'count_band':band,**metric(yy[mask],s[mask],t)})
    pd.DataFrame(band_rows).to_csv(ROOT/'results/count_band_per_seed.csv',index=False)
    summarize(pd.DataFrame(band_rows),['count_band','source_fpr_target'],ROOT/'results/count_band_summary.csv')
    primary=pd.read_csv(ROOT.parent/'results/per_seed_results.csv').query('variant=="prefix2480" and cohort=="external_primary"')
    check=results.query('positive_minimum==5').merge(primary,on=['seed','source_fpr_target'],suffixes=('_new','_old'))
    for m in ['TPR','FPR','ROC_AUC','AP']:assert np.allclose(check[m+'_new'],check[m+'_old'],rtol=0,atol=1e-14),m
    for (_,alpha),g in results.groupby(['seed','source_fpr_target']):assert g.FPR.nunique()==1
    save(ROOT/'logs/existing_score_analyses.json',{'status':'PASS','original_primary_reproduced':True,'FPR_invariance_verified':True,'config_sha256':sha(ROOT/'config/revision_analysis.json'),'code_sha256':sha(Path(__file__))})
    log('existing_score_analyses_complete')

def final():
    cfg,d=metadata();y=d.label.to_numpy();rows=[];related=[]
    near=pd.read_parquet(ROOT/'data/exact_source_relatedness.parquet');assert np.array_equal(near.sha256,d.sha256)
    keep=~near.has_source_neighbor_le30.to_numpy()
    d.assign(excluded=~keep).groupby(['corpus','label','excluded']).size().rename('n').to_csv(ROOT/'results/exact_relatedness_counts.csv')
    st=pd.read_parquet(cfg['inputs']['source_test_metadata']['path']);st=st.loc[st.unique_sha_representative].reset_index(drop=True)
    age=pd.cut(d.report_age_days,[-np.inf,30,365,np.inf],labels=['0-30','31-365','over365'])
    for seed in SEEDS:
        for role in ['source','linear']:assert (ROOT/f'logs/{role}_{seed}_complete.json').exists()
        receipt=read(ROOT/f'logs/linear_{seed}_complete.json');s=np.load(ROOT/f'predictions/linear_external_{seed}.npy');ss=np.load(ROOT/f'predictions/linear_source_test_{seed}.npy')
        for alpha in [.01,.001]:
            t=receipt['thresholds'][str(alpha)];base={'seed':seed,'source_fpr_target':alpha,'threshold':t}
            for group,mask in [('primary',np.ones(len(d),bool)),('exact_TLSH_le30_excluded',keep)]+[(f'DLL_{v}',d.is_dll.eq(v).to_numpy()) for v in [0,1]]+[(f'age_{v}',age.eq(v).to_numpy()) for v in ['0-30','31-365','over365']]+[(f'source_{v}',d.corpus.eq(v).to_numpy()) for v in sorted(d.corpus.unique())]:
                rows.append({**base,'group':group,**metric(y[mask],s[mask],t)})
            rows.append({**base,'group':'source_test',**metric(st.label,ss,t)})
            old=read(ORIGINAL/f'logs/seed_{seed}_complete.json')['variants']['prefix2480'];so=np.load(ROOT.parent/f'predictions/prefix2480/external_seed_{seed}.npy')
            related.append({'seed':seed,'source_fpr_target':alpha,'threshold':old['thresholds'][str(alpha)],**metric(y[keep],so[keep],old['thresholds'][str(alpha)])})
    linear=pd.DataFrame(rows);linear.to_csv(ROOT/'results/linear_per_seed.csv',index=False);summarize(linear,['group','source_fpr_target'],ROOT/'results/linear_summary.csv')
    r=pd.DataFrame(related);r.to_csv(ROOT/'results/exact_relatedness_per_seed.csv',index=False);summarize(r,['source_fpr_target'],ROOT/'results/exact_relatedness_summary.csv')
    diag=pd.concat([pd.read_csv(ROOT/f'results/source_diagnostic_{s}.csv') for s in SEEDS],ignore_index=True);diag.to_csv(ROOT/'results/source_diagnostic_per_seed.csv',index=False);summarize(diag,['group'],ROOT/'results/source_diagnostic_summary.csv',('ROC_AUC','AP','TPR','TNR','balanced_accuracy'))
    old=pd.read_csv(ROOT.parent/'results/per_seed_results.csv').query('variant=="prefix2480" and cohort=="external_primary"')
    joined=linear.query('group=="primary"').merge(old,on=['seed','source_fpr_target'],suffixes=('_linear','_lightgbm'))
    paired=joined[['seed','source_fpr_target']].copy()
    for m in ['TPR','FPR','ROC_AUC','AP']:paired[m+'_difference_pp']=100*(joined[m+'_linear']-joined[m+'_lightgbm'])
    paired.to_csv(ROOT/'results/linear_minus_lightgbm_paired.csv',index=False)
    paired_summary=[]
    for alpha,g in paired.groupby('source_fpr_target'):
        out={'source_fpr_target':alpha,'n_seeds':len(g)}
        for m in ['TPR','FPR','ROC_AUC','AP']:
            out[m+'_difference_pp_mean']=float(g[m+'_difference_pp'].mean())
            out[m+'_difference_pp_sd']=float(g[m+'_difference_pp'].std(ddof=1))
        paired_summary.append(out)
    pd.DataFrame(paired_summary).to_csv(ROOT/'results/linear_minus_lightgbm_paired_summary.csv',index=False)
    save(ROOT/'logs/final_analysis.json',{'status':'COMPLETE','five_source_fits':True,'five_linear_fits':True,'no_primary_retraining':True,'result_sha256':{p.name:sha(p) for p in (ROOT/'results').glob('*.csv')}})
    log('final_revision_analysis_complete')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['existing','final']);a=p.parse_args();(existing if a.phase=='existing' else final)()
