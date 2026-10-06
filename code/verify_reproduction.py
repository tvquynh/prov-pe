"""Independent checks of saved inference and counts; never trains a model."""
import os
os.environ['OMP_NUM_THREADS']='4'
import tempfile
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import *


def main():
    cfg=read(ROOT/'config/experiment.json')
    for name,digest in cfg['code_sha256'].items():
        assert sha(ROOT/'code'/name)==digest
    target=pd.read_parquet(ROOT/'data/evaluation_manifest.parquet')
    # The first 1,024 lexical SHA identifiers are chosen without labels or scores.
    ix=np.argsort(target.sha256.to_numpy())[:1024]
    x=np.load(ROOT/'cache/target.npy',mmap_mode='r')
    all_rows=pd.read_csv(ROOT/'results/per_seed_results.csv')
    near=pd.read_parquet(ROOT/'data/historical_near_pair_ledger.parquet').set_index('sha256').reindex(target.sha256)
    masks={'external_primary':np.ones(len(target),bool),
           'external_temporal':target.temporal_sensitivity.to_numpy(),
           'external_tls_equal':target.identical_tlsh_sensitivity.to_numpy(),
           'external_temporal_tls_equal':target.temporal_and_tlsh_sensitivity.to_numpy(),
           'external_known_near_control':near.primary_without_verified_source_neighbor.to_numpy(bool)}
    source=pd.read_parquet(ROOT/'data/evaluation_source_test.parquet')
    source=source.loc[source.unique_sha_representative].reset_index(drop=True)
    checks=[]
    verified_counts=0
    for seed in SEEDS:
        receipt=read(ROOT/f'logs/seed_{seed}_complete.json')
        for variant,dim in [('prefix2480',2480),('full2568',2568)]:
            path=ROOT/f'models/prefix2480_seed_{seed}.txt' if variant=='prefix2480' else STUDY_A/f'models/baseline_seed_{seed}.txt'
            assert sha(path)==receipt['variants'][variant]['model_sha256']
            model=lgb.Booster(model_file=str(path))
            actual=model.predict(x[ix,:dim],num_iteration=500,num_threads=4)
            stored=np.load(ROOT/f'predictions/{variant}/external_seed_{seed}.npy')
            assert np.array_equal(actual,stored[ix]), ('INFERENCE_MISMATCH',variant,seed)
            checks.append({'seed':seed,'variant':variant,'rows':len(ix),'exact_score_match':True,'model_sha256':sha(path)})
            for cohort in list(masks)+['source_test']:
                if cohort=='source_test':
                    scores=np.load(ROOT/f'predictions/{variant}/source_test_seed_{seed}.npy')
                    y=source.label.to_numpy()
                else:
                    scores=stored[masks[cohort]]
                    y=target.label.to_numpy()[masks[cohort]]
                for alpha in cfg['source_fpr_targets']:
                    threshold=receipt['variants'][variant]['thresholds'][str(alpha)]
                    selected=(scores>=threshold)
                    n0=int(np.count_nonzero(y==0));n1=int(np.count_nonzero(y==1))
                    fp=int(np.count_nonzero(selected & (y==0)))
                    tp=int(np.count_nonzero(selected & (y==1)))
                    r=all_rows[(all_rows.variant==variant)&(all_rows.seed==seed)&(all_rows.cohort==cohort)&(all_rows.source_fpr_target==alpha)].iloc[0]
                    assert (n0,n1,fp,tp)==(r.n_negative,r.n_positive,r.false_positive,r.true_positive)
                    assert abs(fp/n0-r.FPR)<1e-14 and abs(tp/n1-r.TPR)<1e-14
                    verified_counts+=1
    # Re-run the frozen analysis against identical data in an isolated output root.
    with tempfile.TemporaryDirectory(prefix='jisa_reproduction_') as tmp:
        sandbox=Path(tmp)
        # Preserve the original scientific source. Redirect only the output
        # assignment in a transient execution copy, with all other bytes equal.
        source_code=(ROOT/'code/analyze_results.py').read_text('utf8')
        assert source_code.count("output=ROOT/'results'")==1
        patched=source_code.replace("output=ROOT/'results'",'output=Path('+repr(str(sandbox/'results'))+')',1)
        namespace={'__name__':'reproduction_copy','__file__':str(ROOT/'code/analyze_results.py')}
        exec(compile(patched,str(ROOT/'code/analyze_results.py'),'exec'),namespace)
        namespace['main']()
        repeated=[]
        for file in sorted((ROOT/'results').glob('*.csv')):
            copy=sandbox/'results'/file.name
            assert copy.exists() and sha(copy)==sha(file),('METRIC_REPRODUCTION_MISMATCH',file.name)
            repeated.append({'file':file.name,'sha256':sha(file),'byte_identical':True})
    save(ROOT/'qualification/reproduction.json',{'status':'PASS','checks':checks,'count_rows_verified':verified_counts,
          'repeated_csv':repeated,'model_refits':0,'score_subset_selection':'first 1024 lexically ordered target SHA identifiers',
          'scope':'Exact reloaded-model inference on 1024 external rows per model; independent full-cohort confusion counts; byte-identical frozen-analysis CSV regeneration. This is not independent training reproduction.',
          'verification_code_sha256':sha(Path(__file__))})
    log('reproduction_pass',models=len(checks),count_rows=verified_counts,csv_files=len(repeated))


if __name__=='__main__':
    main()
