"""Ten authorized supplementary CPU fits, with separate source diagnostic and detector roles."""
import os
os.environ['OMP_NUM_THREADS']='32'
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import argparse,time,gc,warnings
import numpy as np,pandas as pd,lightgbm as lgb,joblib
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.exceptions import ConvergenceWarning
from revision_common import *

def start(role,seed,cfg):
    done=ROOT/f'logs/{role}_{seed}_complete.json'
    if done.exists():return False
    claim=ROOT/f'logs/{role}_{seed}_claim.json'
    assert not claim.exists(),f'Prior start needs diagnosis: {claim}'
    hashes=read(ROOT/'config/training_code_hashes.json')
    for n,h in hashes.items():assert sha(ROOT/'code'/n)==h
    save(claim,{'utc':datetime.now(timezone.utc).isoformat(),'role':role,'seed':seed,'config_sha256':sha(ROOT/'config/revision_analysis.json'),'code_sha256':hashes})
    return True

def source(seed,cfg):
    if not start('source',seed,cfg):return
    began=time.monotonic()
    d=pd.read_parquet(cfg['inputs']['target_metadata']['path'],columns=['corpus','label'])
    split=np.load(ROOT/f'data/source_diagnostic_split_{seed}.npy')
    assert sha(ROOT/f'data/source_diagnostic_split_{seed}.npy')==cfg['prepared_data_hashes'][f'source_diagnostic_split_{seed}.npy']
    x=np.load(cfg['inputs']['target_features']['path'],mmap_mode='r')
    y=d.corpus.eq('mb_malware_candidate').to_numpy(dtype=np.int8)
    params=read(ORIGINAL/'config/lightgbm.json')
    params.update(num_threads=32,verbosity=-1,seed=seed,bagging_seed=seed,feature_fraction_seed=seed)
    fit=np.flatnonzero(split==0);val=np.flatnonzero(split==1);test=np.flatnonzero(split==2)
    training=lgb.Dataset(x[fit,:2480],label=y[fit],categorical_feature=cfg['categorical_indices'])
    validation=lgb.Dataset(x[val,:2480],label=y[val],categorical_feature=cfg['categorical_indices'],reference=training)
    log('source_fit_started',seed=seed,fit=len(fit),validation=len(val),test=len(test))
    m=lgb.train(params,training,valid_sets=[validation],callbacks=[lgb.log_evaluation(100)])
    path=ROOT/f'models/source_diagnostic_{seed}.txt';m.save_model(str(path))
    del training,validation;gc.collect()
    scores=np.empty(len(test),dtype=np.float64)
    for i in range(0,len(test),20000):scores[i:i+20000]=m.predict(x[test[i:i+20000],:2480],num_threads=32)
    np.savez_compressed(ROOT/f'predictions/source_diagnostic_{seed}.npz',indices=test,scores=scores,labels=y[test])
    rows=[]
    for name,mask in [('all',np.ones(len(test),bool)),('VT_label_0',d.label.to_numpy()[test]==0),('VT_label_1',d.label.to_numpy()[test]==1)]:
        yy=y[test][mask];ss=scores[mask];neg=yy==0;pos=yy==1;pred=ss>=.5
        rows.append({'seed':seed,'group':name,'n_source0':int(neg.sum()),'n_source1':int(pos.sum()),'ROC_AUC':float(roc_auc_score(yy,ss)) if neg.any() and pos.any() else None,'AP':float(average_precision_score(yy,ss)) if pos.any() and neg.any() else None,'TPR':float(pred[pos].mean()) if pos.any() else None,'TNR':float((~pred[neg]).mean()) if neg.any() else None,'balanced_accuracy':float((pred[pos].mean()+(~pred[neg]).mean())/2) if neg.any() and pos.any() else None})
    pd.DataFrame(rows).to_csv(ROOT/f'results/source_diagnostic_{seed}.csv',index=False)
    save(ROOT/f'logs/source_{seed}_complete.json',{'status':'COMPLETE','seed':seed,'elapsed_seconds':time.monotonic()-began,'iterations':m.current_iteration(),'model_sha256':sha(path),'scores_sha256':sha(ROOT/f'predictions/source_diagnostic_{seed}.npz')})
    log('source_fit_complete',seed=seed,seconds=time.monotonic()-began)

def numeric_block(x,idx,cols):
    a=np.asarray(x[idx,:2480],dtype=np.float32)[:,cols].copy()
    np.log1p(np.abs(a),outval:=np.empty_like(a))
    np.copysign(outval,a,out=outval)
    return outval

def transform(x,idx,num,cat,scaler,encoder):
    a=scaler.transform(numeric_block(x,idx,num),copy=False)
    b=encoder.transform(np.asarray(x[idx,:2480])[:,cat])
    z=np.ascontiguousarray(np.concatenate([a,b],axis=1),dtype=np.float32)
    assert np.isfinite(z).all()
    return z

def linear(seed,cfg):
    if not start('linear',seed,cfg):return
    began=time.monotonic()
    x=np.load(cfg['inputs']['source_train_features']['path'],mmap_mode='r')
    y=pd.read_parquet(cfg['inputs']['source_train_metadata']['path'],columns=['label']).label.to_numpy(dtype=np.int32)
    split=np.load(cfg['inputs'][f'split_{seed}']['path'])
    fit=split['fit'];val=split['validation'];cat=cfg['categorical_indices'];num=np.setdiff1d(np.arange(2480),cat)
    scaler=StandardScaler()
    for i in range(0,len(fit),20000):scaler.partial_fit(numeric_block(x,fit[i:i+20000],num))
    encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
    # At most seven categorical columns are materialized here.
    encoder.fit(np.column_stack([x[fit,k] for k in cat]))
    dim=len(num)+sum(len(c) for c in encoder.categories_)
    cache=WORK/'cache'/f'linear_fit_{seed}.npy'
    xx=np.lib.format.open_memmap(cache,mode='w+',dtype=np.float32,shape=(len(fit),dim))
    for i in range(0,len(fit),20000):xx[i:i+20000]=transform(x,fit[i:i+20000],num,cat,scaler,encoder)
    xx.flush()
    spec=cfg['linear_reference']
    m=SGDClassifier(loss='log_loss',penalty='l2',alpha=spec['alpha'],max_iter=spec['max_iter'],tol=spec['tol'],n_iter_no_change=spec['n_iter_no_change'],shuffle=True,average=True,learning_rate='optimal',early_stopping=False,random_state=seed,n_jobs=1,verbose=1)
    log('linear_fit_started',seed=seed,rows=len(fit),features=dim)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always');m.fit(xx,y[fit])
    convergence=[str(w.message) for w in seen if issubclass(w.category,ConvergenceWarning)]
    path=ROOT/f'models/logistic_regression_{seed}.joblib'
    joblib.dump({'model':m,'scaler':scaler,'encoder':encoder,'numeric':num,'categorical':cat,'signed_log1p':True},path)
    del xx;gc.collect()
    assert cache.resolve().is_relative_to((WORK/'cache').resolve());cache.unlink()
    def predict(a,ids):
        result=np.empty(len(ids),dtype=np.float64)
        for i in range(0,len(ids),20000):
            # Logit scores preserve ranking and avoid probability saturation/ties.
            result[i:i+20000]=m.decision_function(transform(a,ids[i:i+20000],num,cat,scaler,encoder))
        assert np.isfinite(result).all();return result
    sv=predict(x,val)
    thresholds={str(alpha):threshold(sv[y[val]==0],alpha) for alpha in [.01,.001]}
    predictions={'validation':sv}
    xt=np.load(cfg['inputs']['target_features']['path'],mmap_mode='r')
    predictions['external']=predict(xt,np.arange(len(xt)))
    st=pd.read_parquet(cfg['inputs']['source_test_metadata']['path'],columns=['unique_sha_representative'])
    xs=np.load(cfg['inputs']['source_test_features']['path'],mmap_mode='r')
    predictions['source_test']=predict(xs,np.flatnonzero(st.unique_sha_representative))
    for name,score in predictions.items():np.save(ROOT/f'predictions/linear_{name}_{seed}.npy',score)
    save(ROOT/f'logs/linear_{seed}_complete.json',{'status':'COMPLETE','seed':seed,'elapsed_seconds':time.monotonic()-began,'iterations':int(m.n_iter_),'optimization_budget_warning':convergence,'transformed_dimensions':dim,'model_sha256':sha(path),'thresholds':thresholds,'validation_negative_n':int((y[val]==0).sum()),'validation_fpr':{str(a):float((sv[y[val]==0]>=thresholds[str(a)]).mean()) for a in [.01,.001]},'scores_sha256':{n:sha(ROOT/f'predictions/linear_{n}_{seed}.npy') for n in predictions}})
    log('linear_fit_complete',seed=seed,seconds=time.monotonic()-began,iterations=int(m.n_iter_))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=['source','linear']);p.add_argument('--seed',type=int);a=p.parse_args()
    cfg=read(ROOT/'config/revision_analysis.json');assert cfg['seeds']==SEEDS and cfg['cpu_only']
    for seed in ([a.seed] if a.seed else SEEDS):
        assert seed in SEEDS
        (source if a.role=='source' else linear)(seed,cfg)
