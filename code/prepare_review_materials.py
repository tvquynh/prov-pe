"""Prepare a compact review artifact and a separate local feature candidate."""
import shutil
import numpy as np
import pandas as pd
from common import *


def copy(source,destination):
    source=Path(source);destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and sha(destination)==sha(source):
        return
    shutil.copy2(source,destination)
    assert sha(source)==sha(destination)


def main():
    bundle=ROOT/'review_package'
    data=bundle/'review_data';data.mkdir(parents=True,exist_ok=True)
    target=pd.read_parquet(ROOT/'data/evaluation_manifest.parquet')
    near=pd.read_parquet(ROOT/'data/historical_near_pair_ledger.parquet').set_index('sha256').reindex(target.sha256)
    columns=['sha256','label','file_type','corpus','is_dll','report_age_days','target_row',
             'temporal_sensitivity','identical_tlsh_sensitivity','temporal_and_tlsh_sensitivity']
    minimal=target[columns].copy()
    minimal['known_near_control']=near.primary_without_verified_source_neighbor.to_numpy(bool)
    minimal.to_parquet(data/'target.parquet',index=False,compression='zstd')
    source=pd.read_parquet(ROOT/'data/evaluation_source_test.parquet')
    source.loc[source.unique_sha_representative,['sha256','label']].reset_index(drop=True).to_parquet(data/'source_test.parquet',index=False,compression='zstd')
    inventory=pd.read_parquet(ROOT/'data/target_manifest.parquet')
    inventory[['sha256','label','file_type','corpus','eligible_type','eligible_label','eligible_time','overlap_source_union','primary',
               'temporal_sensitivity','identical_tlsh_sensitivity','temporal_and_tlsh_sensitivity']].to_parquet(data/'inventory_eligibility.parquet',index=False,compression='zstd')
    ix=np.argsort(target.sha256.to_numpy())[:1024]
    x=np.load(ROOT/'cache/target.npy',mmap_mode='r')
    np.savez_compressed(data/'inference_fixture.npz',indices=ix,features=x[ix],sha256=target.sha256.to_numpy(dtype='U64')[ix])
    train_labels=pd.read_parquet(ROOT/'data/evaluation_source_train.parquet',columns=['label']).label.to_numpy()
    for seed in SEEDS:
        split=np.load(STUDY_A/f'data/internal_split_{seed}.npz')
        np.save(data/f'validation_labels_{seed}.npy',train_labels[split['validation']])
        copy(ROOT/f'models/prefix2480_seed_{seed}.txt',bundle/f'models/prefix2480_seed_{seed}.txt')
        copy(STUDY_A/f'models/baseline_seed_{seed}.txt',bundle/f'models/full2568_seed_{seed}.txt')
    for folder in ['config','qualification','results','logs','code']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                copy(path,bundle/path.relative_to(ROOT))
    for path in (ROOT/'predictions').rglob('*.npy'):
        copy(path,bundle/path.relative_to(ROOT))
    history={
        'extractor_PARITY_REPORT.json':Path(r'E:\phase3\OPS_EMBER_FEATURES_20260915\PARITY_REPORT.json'),
        'extractor_VERIFY_REPORT.json':Path(r'E:\phase3\OPS_EMBER_FEATURES_20260915\VERIFY_REPORT.json'),
        'extractor_RUN_SUMMARY.json':Path(r'E:\phase3\OPS_EMBER_FEATURES_20260915\RUN_SUMMARY.json'),
        'extract_features.py':Path(r'E:\phase3\OPS_EMBER_FEATURES_20260915\extract_features.py'),
        'jsonl_VALIDATION_REPORT.json':DATASET/'jsonl_vt_20260927/metadata/VALIDATION_REPORT.json',
        'parquet_VALIDATION_REPORT.json':DATASET/'parquet_vt_20260927/VALIDATION_REPORT.json',
        'build_jsonl_vt_20260927.py':DATASET/'code/build_jsonl_vt_20260927.py',
        'metadata_audit.sql':ROOT.parent/'review/metadata_audit.sql'}
    history_id={}
    for name,path in history.items():
        assert path.exists(),path
        copy(path,bundle/'historical_validation'/name)
        history_id[name]={'original_path':str(path),'sha256':sha(path),'role':'Retained historical source, not a new execution receipt'}
    save(bundle/'historical_validation/source_identity.json',history_id)
    save(data/'identities.json',{'parent_target_manifest_sha256':sha(ROOT/'data/evaluation_manifest.parquet'),
        'parent_source_test_manifest_sha256':sha(ROOT/'data/evaluation_source_test.parquet'),
        'files':{str(p.relative_to(bundle)).replace('\\','/'):sha(p) for folder in ['review_data','models','predictions'] for p in (bundle/folder).rglob('*') if p.is_file() and p.name!='identities.json'}})
    # This is a local candidate, not a claim of public deposition or licensing.
    resource=ROOT/'feature_resource_candidate'
    for name in ['collected_pe_ember_v3.parquet','sample_provenance.parquet']:
        copy(DATASET/'parquet_vt_20260927'/name,resource/name)
    for name in ['target_manifest.parquet','historical_near_pair_ledger.parquet']:
        copy(ROOT/'data'/name,resource/name)
    for path in (ROOT/'config').glob('*.json'):
        copy(path,resource/'config'/path.name)
    save(resource/'RESOURCE_STATUS.json',{'status':'LOCAL_FEATURE_CANDIDATE_PREPARED','public_deposit':False,'public_license':None,
         'not_included':['original PE binaries','raw VirusTotal responses','credentials'],
         'files':{str(p.relative_to(resource)).replace('\\','/'):sha(p) for p in resource.rglob('*') if p.is_file() and p.name!='RESOURCE_STATUS.json'}})
    save(bundle/'FEATURE_RESOURCE_POINTER.json',{'local_path':str(resource),'status':'LOCAL_CANDIDATE_NOT_PUBLIC_DEPOSIT',
         'resource_status_sha256':sha(resource/'RESOURCE_STATUS.json'),'resource_files':read(resource/'RESOURCE_STATUS.json')['files']})
    log('review_materials_prepared',directory=str(bundle))


if __name__=='__main__':
    main()
