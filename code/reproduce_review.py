"""Portable review reproduction from enclosed scores, metadata and model fixture.

This entry point never contacts VT, executes a PE file or trains a model.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import platform
import time
from cas_revision_common import require, fresh, environment, save
from fixture_profiles import compare_fixture


def load_dependencies():
    global np, pd, lgb, roc_auc_score, average_precision_score
    import numpy as np
    import pandas as pd
    import pyarrow
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def observed_environment(packages, output):
    """Record runtime observations without claiming unmeasured isolation."""
    runtime = environment(packages)
    runtime.update(python_prefix=sys.prefix, python_base_prefix=sys.base_prefix,
                   prefix_differs_from_base=sys.prefix != sys.base_prefix,
                   output_directory=str(Path(output).resolve()))
    return runtime


def main(root, output, profile, report):
    root=Path(root).resolve()
    identity=json.loads((root/'review_data/identities.json').read_text('utf8'))
    for rel,expected in identity['files'].items():
        require(digest(root/rel)==expected, 'IDENTITY_MISMATCH: '+rel)
    target=pd.read_parquet(root/'review_data/target.parquet')
    source=pd.read_parquet(root/'review_data/source_test.parquet')
    fixture=np.load(root/'review_data/inference_fixture.npz', allow_pickle=False)
    indices=fixture['indices']
    require(indices.shape==(1024,) and fixture['features'].shape==(1024,2568), 'FIXTURE_SHAPE')
    require(np.issubdtype(indices.dtype,np.integer) and (indices>=0).all() and (indices<len(target)).all(), 'FIXTURE_INDEX_RANGE')
    require(np.unique(indices).size==1024 and target.sha256.is_unique and source.sha256.is_unique, 'ROW_ID_UNIQUENESS')
    require(np.array_equal(target.iloc[indices].sha256.to_numpy(),fixture['sha256']), 'FIXTURE_SHA_LINKAGE')
    require(np.array_equal(indices,np.argsort(target.sha256.to_numpy(),kind='stable')[:1024]), 'FIXTURE_SELECTION_RULE')
    require(np.isfinite(fixture['features']).all(), 'NONFINITE_FIXTURE')
    report['frozen_identity_files_verified']=len(identity['files'])
    expected=pd.read_csv(root/'results/per_seed_results.csv')
    cohorts={'external_primary':np.ones(len(target),bool),'external_temporal':target.temporal_sensitivity.to_numpy(),
             'external_tls_equal':target.identical_tlsh_sensitivity.to_numpy(),
             'external_temporal_tls_equal':target.temporal_and_tlsh_sensitivity.to_numpy(),
             'external_known_near_control':target.known_near_control.to_numpy()}
    rows=[]
    checks=report['checks']
    for seed in [2026,2027,2028,2029,2030]:
        receipt=json.loads((root/f'logs/seed_{seed}_complete.json').read_text('utf8'))
        yval=np.load(root/f'review_data/validation_labels_{seed}.npy')
        for variant,dim in [('prefix2480',2480),('full2568',2568)]:
            model=lgb.Booster(model_file=str(root/f'models/{variant}_seed_{seed}.txt'))
            external=np.load(root/f'predictions/{variant}/external_seed_{seed}.npy')
            test=np.load(root/f'predictions/{variant}/source_test_seed_{seed}.npy')
            validation=np.load(root/f'predictions/{variant}/validation_seed_{seed}.npy')
            require(external.shape==(len(target),) and test.shape==(len(source),) and validation.shape==yval.shape, 'SCORE_ROW_COUNTS')
            require(all(np.isfinite(x).all() for x in [external,test,validation]), 'NONFINITE_SAVED_SCORES')
            require(np.isin(yval,[0,1]).all(), 'VALIDATION_LABEL_DOMAIN')
            report['model_inference_calls']+=1
            predicted=model.predict(fixture['features'][:,:dim],num_iteration=500,num_threads=4)
            check=compare_fixture(predicted,external[indices],receipt['variants'][variant]['thresholds'],profile)
            check.update(seed=seed,variant=variant,rows=1024)
            checks.append(check)
            report['operating_point_comparisons']+=len(check['decision_disagreements'])
            require(check['status']=='PASS', 'FIXTURE_PROFILE_FAILED: '+str(check))
            negative=np.sort(validation[yval==0])
            distinct=np.unique(negative)
            fp_counts=len(negative)-np.searchsorted(negative,distinct,side='left')
            for alpha in [.01,.001]:
                valid=distinct[fp_counts<=int(np.floor(alpha*len(negative)))]
                threshold=float(valid[0]) if len(valid) else float(np.nextafter(negative[-1],np.inf))
                require(threshold==receipt['variants'][variant]['thresholds'][str(alpha)], 'FROZEN_THRESHOLD_RECONSTRUCTION')
                report['thresholds_reconstructed']+=1
                for cohort in list(cohorts)+['source_test']:
                    mask=cohorts.get(cohort)
                    y=source.label.to_numpy() if mask is None else target.label.to_numpy()[mask]
                    s=test if mask is None else external[mask]
                    positive=y==1;negative_class=y==0;decision=s>=threshold
                    n0=int(negative_class.sum());n1=int(positive.sum())
                    fp=int(decision[negative_class].sum());tp=int(decision[positive].sum())
                    row={'variant':variant,'seed':seed,'source_fpr_target':alpha,'cohort':cohort,
                         'threshold':threshold,'n_negative':n0,'n_positive':n1,'false_positive':fp,'true_positive':tp,
                         'FPR':fp/n0,'TPR':tp/n1,'ROC_AUC':roc_auc_score(y,s),'AP':average_precision_score(y,s)}
                    selected=expected[(expected.variant==variant)&(expected.seed==seed)&(expected.source_fpr_target==alpha)&(expected.cohort==cohort)]
                    require(len(selected)==1, 'RESULT_ROW_UNIQUENESS')
                    check=selected.iloc[0]
                    for key in ['threshold','n_negative','n_positive','false_positive','true_positive','FPR','TPR','ROC_AUC','AP']:
                        if key in ['n_negative','n_positive','false_positive','true_positive']:
                            require(row[key]==check[key], 'INTEGER_COUNT_MISMATCH: '+str((key,cohort,seed,variant)))
                        else:
                            require(abs(row[key]-check[key])<1e-12, 'METRIC_MISMATCH: '+str((key,cohort,seed,variant)))
                    rows.append(row)

        print(f'Seed {seed}: inference, thresholds and metrics verified',flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'validated_per_seed_results.csv',index=False)
    summaries=pd.read_csv(root/'results/summary_results.csv')
    for _,row in summaries.iterrows():
        group=frame[(frame.variant==row.variant)&(frame.cohort==row.cohort)&(frame.source_fpr_target==row.source_fpr_target)]
        require(len(group)==5, 'FIVE_SEED_SUMMARY_REQUIRED')
        for metric in ['TPR','FPR','ROC_AUC','AP']:
            require(abs(group[metric].mean()-row[metric+'_mean'])<1e-12, 'MEAN_MISMATCH')
            require(abs(group[metric].std(ddof=1)-row[metric+'_sd'])<1e-12, 'SAMPLE_SD_MISMATCH')
    report.update(status='PASS', per_seed_rows=len(rows), summary_rows=len(summaries),
                  bitwise_exact_models=sum(x['exact_bitwise'] for x in checks),
                  max_fixture_absolute_error=max(x['max_absolute_error'] for x in checks),
                  metric_tolerance=1e-12, integer_counts_exact=True, full_training_reproduction=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',choices=['exact','numerical'],required=True,help='No default: select explicitly.')
    args=parser.parse_args();args.root=args.root.resolve(strict=True)
    output=fresh(args.output,[args.root])
    started=time.perf_counter()
    packages=['numpy','pandas','pyarrow','lightgbm','scikit-learn','scipy']
    report={'status':'STARTED','profile':args.profile,'default_profile':None,
            'environment':observed_environment(packages, output),'checks':[], 'training_runs':0,
            'model_inference_calls':0,'operating_point_comparisons':0,'thresholds_reconstructed':0,
            'full_corpus_inference_runs':0,'VT_requests':0,'PE_parses':0,'bootstrap_runs':0,'new_experiments':0,
            'fixture_tolerance_rationale':'Fixed absolute 1e-12 for probabilities, relative zero, and zero allowed decision changes. Counts and frozen thresholds remain exact; original metric tolerance remains strictly less than 1e-12.'}
    exit_code=0
    try:
        load_dependencies()
        pins={}
        for line in (args.root/'environment/pip-freeze.txt').read_text().splitlines():
            if '==' in line:
                key,value=line.split('==',1)
                if key in packages:pins[key]=value
        matches={key:report['environment']['versions'][key]==pins.get(key) for key in packages}
        matches['python']=platform.python_version()=='3.11.9'
        report['environment'].update(recorded_software_pins=pins,software_pin_matches=matches)
        if args.profile=='exact' and not all(matches.values()):
            raise ImportError('Exact profile requires recorded software versions: '+str(matches))
    except (ImportError,OSError) as error:
        report.update(status='ENVIRONMENT_BLOCKED',error=str(error));exit_code=2
    else:
        try:main(args.root,output,args.profile,report)
        except Exception as error:
            report.update(status='SCIENTIFIC_CHECK_FAILED',error=repr(error));exit_code=1
    report['elapsed_seconds']=time.perf_counter()-started
    save(output/'review_reproduction.json',report)
    print(json.dumps({k:v for k,v in report.items() if k in ['status','error','model_inference_calls','operating_point_comparisons','thresholds_reconstructed','per_seed_rows','summary_rows','bitwise_exact_models','max_fixture_absolute_error','elapsed_seconds']},indent=2),flush=True)
    sys.exit(exit_code)
