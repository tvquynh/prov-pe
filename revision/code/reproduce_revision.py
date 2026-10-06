"""Portable saved-score and small-fixture verification; never fits a classifier.

Requires the complete compact candidate, but no private feature matrices, VT
access, raw executables or source training data. Output must be a new directory.
"""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
import argparse, json, time, platform, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import tlsh
from sklearn.metrics import roc_auc_score, average_precision_score
from revision_common import ROOT, SEEDS, read, save, sha, threshold
from analyze_revision import metric, summarize

def require(condition, message):
    if not condition: raise RuntimeError(message)

def compare(a,b,columns,tol=1e-12):
    require(len(a)==len(b),'row count mismatch')
    for col in columns:
        x=pd.to_numeric(a[col]).to_numpy(float);y=pd.to_numeric(b[col]).to_numpy(float)
        require(np.array_equal(np.isnan(x),np.isnan(y)),f'undefined metric mismatch: {col}')
        require(np.allclose(x,y,rtol=0,atol=tol,equal_nan=True),f'metric mismatch: {col}')

def linear_transform(features,bundle):
    # Independent implementation for inference on the included fixture.
    x=np.array(features[:,bundle['numeric']],dtype=np.float32,copy=True)
    x=np.sign(x)*np.log1p(np.abs(x))
    x=(x-bundle['scaler'].mean_).astype(np.float32)
    x=(x/bundle['scaler'].scale_).astype(np.float32)
    categorical=bundle['encoder'].transform(features[:,bundle['categorical']])
    return np.ascontiguousarray(np.column_stack((x,categorical)),dtype=np.float32)

def fixture_check(observed,expected,thresholds,atol):
    require(observed.shape==expected.shape,'fixture shape mismatch')
    require(np.isfinite(observed).all() and np.isfinite(expected).all(),'nonfinite fixture')
    error=float(np.max(np.abs(observed-expected)))
    require(error<=atol,'fixture numerical error')
    disagreements={str(t):int(np.count_nonzero((observed>=t)!=(expected>=t))) for t in thresholds}
    require(not any(disagreements.values()),'fixture threshold decision changed')
    return {'rows':len(observed),'max_absolute_error':error,'tolerance':atol,'decision_disagreements':disagreements}

def run(output):
    began=time.monotonic();output=Path(output).resolve()
    require(not output.exists(),'Output already exists; choose a fresh directory')
    output.mkdir(parents=True)
    report={'status':'RUNNING','started_utc':pd.Timestamp.now(tz='UTC').isoformat(),'scope':'Saved-score analysis, threshold reconstruction and compact inference fixtures; no training or full source-distance rerun','environment':{'python':platform.python_version(),'platform':platform.platform(),**{k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','lightgbm','pyarrow','joblib']}}}
    try:
        if (ROOT/'config/compact_identities.json').exists():
            identities=read(ROOT/'config/compact_identities.json')['files']
            for name,h in identities.items():require(sha(ROOT/name)==h,'identity mismatch: '+name)
            report['identity_files_verified']=len(identities)
        else:
            raise RuntimeError('Missing compact identity manifest')
        base=ROOT.parent;d=pd.read_parquet(ROOT/'data/revision_target_metadata.parquet');y=d.label.to_numpy()
        original=pd.read_parquet(base/'review_data/target.parquet')
        require(np.array_equal(d.sha256,original.sha256) and np.array_equal(y,original.label),'primary linkage mismatch')
        near=pd.read_parquet(ROOT/'data/exact_source_relatedness.parquet')
        require(np.array_equal(near.sha256,d.sha256),'relatedness linkage mismatch')
        retained=~near.has_source_neighbor_le30.to_numpy()
        require((near.loc[~retained,'witness_distance']<=30).all(),'witness exceeds distance threshold')
        for a,b,dist in near.loc[~retained,['tlsh_canonical','reference_tlsh','witness_distance']].itertuples(index=False,name=None):
            require(tlsh.diff(a,b)==dist,'original-library witness disagreement')
        report['library_witnesses_verified']=int((~retained).sum())
        weights=np.load(ROOT/'data/common_support_weights.npy')
        require(len(weights)==len(d) and np.isfinite(weights).all() and (weights>=0).all(),'invalid support weights')
        cells=pd.read_csv(ROOT/'data/common_support_cell_counts.csv').set_index(['label','cell'])
        sources=sorted(d.corpus.unique());mins=cells[sources].min(axis=1)
        min_lookup=mins.to_dict()
        supported=np.array([min_lookup[(l,c)]>=20 for l,c in zip(d.label,d.cell)])
        require(np.array_equal(supported,d.common_support),'common support membership mismatch')
        rebuilt=np.zeros(len(d))
        for label in [0,1]:
            good=(mins>=20)&(mins.index.get_level_values('label')==label);total=float(mins[good].sum())
            for cell,g in d.loc[supported & d.label.eq(label)].groupby('cell'):
                mass=float(mins.loc[(label,cell)])/total
                for source in sources:
                    ix=g.index[g.corpus.eq(source)];rebuilt[ix]=mass/(2*len(ix))
            rebuilt[d.label.eq(label)]*=int((supported & d.label.eq(label)).sum())
        require(np.allclose(weights,rebuilt,rtol=0,atol=1e-12),'standardization weights mismatch')
        st=pd.read_parquet(base/'review_data/source_test.parquet')
        fixture=np.load(base/'review_data/inference_fixture.npz',allow_pickle=False);ix=fixture['indices'];fx=fixture['features'][:,:2480]
        require(np.array_equal(fixture['sha256'],d.sha256.to_numpy()[ix]),'fixture SHA linkage mismatch')
        age=pd.cut(d.report_age_days,[-np.inf,30,365,np.inf],labels=['0-30','31-365','over365'])
        bands=pd.cut(d.malicious_count,[4,9,14,29,np.inf],labels=['5-9','10-14','15-29','30+'])
        label_rows=[];age_rows=[];band_rows=[];support_rows=[];near_rows=[];linear_rows=[];source_rows=[];inference=[];thresholds=[]
        for seed in SEEDS:
            old=read(base/f'logs/seed_{seed}_complete.json')['variants']['prefix2480'];s=np.load(base/f'predictions/prefix2480/external_seed_{seed}.npy')
            linear=read(ROOT/f'logs/linear_{seed}_complete.json');sl=np.load(ROOT/f'predictions/linear_external_{seed}.npy');ss=np.load(ROOT/f'predictions/linear_source_test_{seed}.npy')
            sv=np.load(ROOT/f'predictions/linear_validation_{seed}.npy');yv=np.load(base/f'review_data/validation_labels_{seed}.npy')
            require(np.isfinite(sl).all() and np.isfinite(ss).all() and np.isfinite(sv).all(),'nonfinite linear scores')
            bundle=joblib.load(ROOT/f'models/logistic_regression_{seed}.joblib')
            observed=bundle['model'].decision_function(linear_transform(fx,bundle))
            inference.append({'seed':seed,'role':'linear',**fixture_check(observed,sl[ix],list(linear['thresholds'].values()),1e-4)})
            for alpha in [.01,.001]:
                t=old['thresholds'][str(alpha)];common={'seed':seed,'source_fpr_target':alpha,'threshold':t}
                computed=threshold(sv[yv==0],alpha);require(computed==linear['thresholds'][str(alpha)],'linear threshold mismatch')
                thresholds.append({'seed':seed,'source_fpr_target':alpha,'threshold':computed,'validation_fpr':float((sv[yv==0]>=computed).mean())})
                for k in [5,10,15]:
                    keep=(y==0)|d.malicious_count.ge(k).to_numpy()
                    label_rows.append({**common,'positive_minimum':k,**metric(y[keep],s[keep],t)})
                    for a in ['0-30','31-365','over365']:
                        mask=keep&age.eq(a).to_numpy();age_rows.append({**common,'positive_minimum':k,'report_age':a,**metric(y[mask],s[mask],t)})
                for b in ['5-9','10-14','15-29','30+']:
                    mask=bands.eq(b).to_numpy();band_rows.append({**common,'count_band':b,**metric(y[mask],s[mask],t)})
                for view,mask,w in [('primary',np.ones(len(d),bool),None),('restricted_unweighted',supported,None),('source_standardized',supported,weights[supported])]:
                    support_rows.append({**common,'view':view,'source':'pooled',**metric(y[mask],s[mask],t,w)})
                for source in sources:
                    mask=supported&d.corpus.eq(source).to_numpy();support_rows.append({**common,'view':'source_standardized','source':source,**metric(y[mask],s[mask],t,weights[mask])})
                near_rows.append({**common,**metric(y[retained],s[retained],t)})
                cl={'seed':seed,'source_fpr_target':alpha,'threshold':computed}
                for group,mask in [('primary',np.ones(len(d),bool)),('exact_TLSH_le30_excluded',retained)]+[(f'DLL_{v}',d.is_dll.eq(v).to_numpy()) for v in [0,1]]+[(f'age_{v}',age.eq(v).to_numpy()) for v in ['0-30','31-365','over365']]+[(f'source_{v}',d.corpus.eq(v).to_numpy()) for v in sources]:
                    linear_rows.append({**cl,'group':group,**metric(y[mask],sl[mask],computed)})
                linear_rows.append({**cl,'group':'source_test',**metric(st.label,ss,computed)})
            src=np.load(ROOT/f'predictions/source_diagnostic_{seed}.npz');indices=src['indices'];scores=src['scores'];sy=src['labels']
            split=np.load(ROOT/f'data/source_diagnostic_split_{seed}.npy')
            require(np.array_equal(indices,np.flatnonzero(split==2)),'source test membership mismatch')
            require(np.array_equal(sy,d.corpus.eq('mb_malware_candidate').to_numpy()[indices]),'source labels mismatch')
            require(pd.DataFrame({'tlsh':d.tlsh_canonical,'split':split}).groupby('tlsh').split.nunique().max()==1,'TLSH group split leakage')
            # The label-blind original fixture is intersected with each held-out source test.
            loc=np.searchsorted(indices,ix);valid=(loc<len(indices));valid[valid]&=(indices[loc[valid]]==ix[valid])
            model=lgb.Booster(model_file=str(ROOT/f'models/source_diagnostic_{seed}.txt'))
            obs=model.predict(fx[valid],num_threads=4)
            inference.append({'seed':seed,'role':'source',**fixture_check(obs,scores[loc[valid]],[.5],1e-12)})
            for name,mask in [('all',np.ones(len(indices),bool)),('VT_label_0',y[indices]==0),('VT_label_1',y[indices]==1)]:
                yy=sy[mask];sp=scores[mask];z=yy==0;o=yy==1;p=sp>=.5
                source_rows.append({'seed':seed,'group':name,'n_source0':int(z.sum()),'n_source1':int(o.sum()),'ROC_AUC':roc_auc_score(yy,sp),'AP':average_precision_score(yy,sp),'TPR':p[o].mean(),'TNR':(~p[z]).mean(),'balanced_accuracy':(p[o].mean()+(~p[z]).mean())/2})
        outputs={'label_sensitivity_per_seed':label_rows,'label_by_age_per_seed':age_rows,'count_band_per_seed':band_rows,'common_support_per_seed':support_rows,'exact_relatedness_per_seed':near_rows,'linear_per_seed':linear_rows,'source_diagnostic_per_seed':source_rows}
        summaries={'label_sensitivity':(['positive_minimum','source_fpr_target']), 'label_by_age':(['positive_minimum','report_age','source_fpr_target']),'count_band':(['count_band','source_fpr_target']),'common_support':(['view','source','source_fpr_target']),'exact_relatedness':(['source_fpr_target']),'linear':(['group','source_fpr_target']),'source_diagnostic':(['group'])}
        report['result_conditions_verified']=0
        for name,rows in outputs.items():
            actual=pd.DataFrame(rows);expected=pd.read_csv(ROOT/f'results/{name}.csv')
            keys=[k for k in ['seed','source_fpr_target','positive_minimum','report_age','count_band','view','source','group'] if k in actual]
            actual=actual.sort_values(keys).reset_index(drop=True);expected=expected.sort_values(keys).reset_index(drop=True)
            require(actual[keys].equals(expected[keys]),'condition identity mismatch: '+name)
            require(set(expected.columns)<=set(actual.columns),'missing reconstructed field: '+name)
            require(set(actual.columns)-set(expected.columns)<={'threshold'},'unexpected reconstructed field: '+name)
            cols=[c for c in expected if c not in keys];compare(actual,expected,cols)
            actual.to_csv(output/f'{name}.csv',index=False);report['result_conditions_verified']+=len(actual)
            stem=name.removesuffix('_per_seed');metrics=('ROC_AUC','AP','TPR','TNR','balanced_accuracy') if stem=='source_diagnostic' else ('TPR','FPR','ROC_AUC','AP')
            summarize(actual,summaries[stem],output/f'{stem}_summary.csv',metrics)
            sa=pd.read_csv(output/f'{stem}_summary.csv');se=pd.read_csv(ROOT/f'results/{stem}_summary.csv');compare(sa,se,[c for c in sa if c not in summaries[stem]])
        pd.DataFrame(thresholds).to_csv(output/'linear_thresholds.csv',index=False)
        original_primary=pd.read_csv(base/'results/per_seed_results.csv').query('variant=="prefix2480" and cohort=="external_primary"')
        joined=pd.DataFrame(linear_rows).query('group=="primary"').merge(original_primary,on=['seed','source_fpr_target'],suffixes=('_linear','_lightgbm'))
        paired=joined[['seed','source_fpr_target']].copy()
        for m in ['TPR','FPR','ROC_AUC','AP']:paired[m+'_difference_pp']=100*(joined[m+'_linear']-joined[m+'_lightgbm'])
        expected=pd.read_csv(ROOT/'results/linear_minus_lightgbm_paired.csv')
        compare(paired,expected,[c for c in paired if c not in ['seed','source_fpr_target']])
        paired.to_csv(output/'linear_minus_lightgbm_paired.csv',index=False)
        expected_summary=pd.read_csv(ROOT/'results/linear_minus_lightgbm_paired_summary.csv')
        for alpha,g in paired.groupby('source_fpr_target'):
            expected_row=expected_summary.loc[expected_summary.source_fpr_target.eq(alpha)].iloc[0]
            for m in ['TPR','FPR','ROC_AUC','AP']:
                for f,fn in [('mean',lambda x:x.mean()),('sd',lambda x:x.std(ddof=1))]:
                    require(abs(fn(g[m+'_difference_pp'])-expected_row[m+'_difference_pp_'+f])<1e-12,'paired difference summary mismatch')
        report['paired_difference_conditions_verified']=len(paired)
        save(output/'inference_checks.json',inference)
        report.update(status='PASS',inference_checks=inference,thresholds_verified=len(thresholds),elapsed_seconds=time.monotonic()-began,finished_utc=pd.Timestamp.now(tz='UTC').isoformat(),checker_sha256=sha(Path(__file__)))
    except Exception as e:
        report.update(status='FAIL',error=repr(e),elapsed_seconds=time.monotonic()-began);save(output/'verification.json',report);raise
    save(output/'verification.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['inference_checks','environment']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
