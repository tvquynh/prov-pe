"""Five fixed CPU fits; source-only thresholds; immutable scores and receipts."""
import os
os.environ['OMP_NUM_THREADS']='48'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse
import gc
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import *


def threshold(negative_scores, alpha):
    scores = np.asarray(negative_scores, dtype=np.float64)
    assert scores.ndim == 1 and len(scores) and np.isfinite(scores).all()
    unique, counts = np.unique(scores, return_counts=True)
    false_positives = np.cumsum(counts[::-1])[::-1]
    allowed = int(np.floor(alpha * len(scores)))
    valid = np.flatnonzero(false_positives <= allowed)
    return float(unique[valid[0]]) if len(valid) else float(np.nextafter(unique[-1], np.inf))


def predict_chunks(model, x, dimensions, indices=None):
    n = len(x) if indices is None else len(indices)
    result = np.empty(n, dtype=np.float64)
    for start in range(0,n,25000):
        batch = x[start:start+25000,:dimensions] if indices is None else x[indices[start:start+25000],:dimensions]
        result[start:start+len(batch)] = model.predict(batch, num_iteration=500, num_threads=48)
    assert np.isfinite(result).all() and ((result>=0)&(result<=1)).all()
    return result


def run(seed):
    cfg = read(ROOT / 'config/experiment.json')
    assert cfg['status']=='FROZEN_BEFORE_NEW_TARGET_PERFORMANCE' and seed in cfg['seeds']
    for name, digest in cfg['code_sha256'].items():
        assert sha(ROOT/'code'/name) == digest, ('FROZEN_CODE_CHANGED', name)
    params = read(ROOT / 'config/lightgbm.json')
    assert sha(ROOT/'config/lightgbm.json') == cfg['lightgbm_parameters_sha256']
    for key in ['seed','bagging_seed','feature_fraction_seed']:
        params[key] = seed
    claim = ROOT / f'logs/seed_{seed}_claim.json'
    done = ROOT / f'logs/seed_{seed}_complete.json'
    if done.exists():
        log('already_complete_skipped',seed=seed)
        return
    assert not claim.exists(), 'A started seed requires explicit technical diagnosis before resume; do not silently refit'
    claim.parent.mkdir(parents=True,exist_ok=True)
    save(claim, {'seed':seed,'experiment_sha256':sha(ROOT/'config/experiment.json'),'parameters':params})
    started = time.monotonic()
    split = np.load(STUDY_A/f'data/internal_split_{seed}.npz')
    labels = pd.read_parquet(ROOT/'data/evaluation_source_train.parquet',columns=['label']).label.to_numpy()
    x = np.load(CACHE/'train.npy',mmap_mode='r')
    log('fit_started',seed=seed,features=2480,fit_rows=len(split['fit']))
    training = lgb.Dataset(x[split['fit'],:2480], label=labels[split['fit']], categorical_feature=cfg['categorical_indices'])
    validation = lgb.Dataset(x[split['validation'],:2480], label=labels[split['validation']], categorical_feature=cfg['categorical_indices'],reference=training)
    model = lgb.train(params, training, valid_sets=[validation], callbacks=[lgb.log_evaluation(100)])
    assert model.current_iteration()==500
    (ROOT/'models').mkdir(exist_ok=True)
    model_path = ROOT/f'models/prefix2480_seed_{seed}.txt'
    model.save_model(str(model_path))
    del training, validation
    gc.collect()
    test_meta = pd.read_parquet(ROOT/'data/evaluation_source_test.parquet')
    test_indices = np.flatnonzero(test_meta.unique_sha_representative.to_numpy())
    records = {}
    for variant, dimensions in [('prefix2480',2480),('full2568',2568)]:
        if variant=='full2568':
            model_path = STUDY_A/f'models/baseline_seed_{seed}.txt'
            model = lgb.Booster(model_file=str(model_path))
        scores_val = predict_chunks(model,x,dimensions,split['validation'])
        thresholds = {str(alpha):threshold(scores_val[labels[split['validation']]==0],alpha) for alpha in cfg['source_fpr_targets']}
        output = ROOT/'predictions'/variant
        output.mkdir(parents=True,exist_ok=True)
        np.save(output/f'validation_seed_{seed}.npy',scores_val)
        test = np.load(CACHE/'test.npy',mmap_mode='r')
        scores_test = predict_chunks(model,test,dimensions,test_indices)
        np.save(output/f'source_test_seed_{seed}.npy',scores_test)
        target = np.load(ROOT/'cache/target.npy',mmap_mode='r')
        scores_target = predict_chunks(model,target,dimensions)
        np.save(output/f'external_seed_{seed}.npy',scores_target)
        records[variant] = {'model_sha256':sha(model_path),'thresholds':thresholds,
                            'validation_negative_n':int((labels[split['validation']]==0).sum()),
                            'validation_actual_fpr':{alpha:float((scores_val[labels[split['validation']]==0]>=value).mean()) for alpha,value in thresholds.items()},
                            'scores_sha256':{key:sha(output/f'{key}_seed_{seed}.npy') for key in ['validation','source_test','external']}}
        del test,target,scores_test,scores_target,scores_val
        log('prediction_complete',seed=seed,variant=variant)
    save(done,{'seed':seed,'elapsed_seconds':time.monotonic()-started,'experiment_sha256':sha(ROOT/'config/experiment.json'),'variants':records})
    log('seed_complete',seed=seed,elapsed_seconds=time.monotonic()-started)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    for seed in ([args.seed] if args.seed is not None else SEEDS):
        run(seed)
