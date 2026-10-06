"""Freeze the fixed evaluation specification before any target predictions."""
import sys
import platform
import importlib.metadata
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common import *


def main():
    assert not (ROOT / 'config/experiment.json').exists()
    audit = read(ROOT / 'qualification/input_audit.json')
    diagnostic = read(ROOT / 'qualification/warning_diagnostic.json')
    assert audit['status'] == 'PASS' and audit['no_model_scores_computed']
    assert diagnostic['same_hash_seed_repeated_identically']
    assert diagnostic['all_observed_differences_confined_to_warning_group']
    assert diagnostic['observed_distinct_warning_representations'] > 1
    manifest = pd.read_parquet(ROOT / 'data/target_manifest.parquet')
    assert sha(ROOT / 'data/target_manifest.parquet') == audit['target_manifest_sha256']
    columns = [f'feature_{i:04d}' for i in range(2568)]
    path = DATASET / 'parquet_vt_20260927/collected_pe_ember_v3.parquet'
    parquet = pq.ParquetFile(path)
    assert [n for n in parquet.schema_arrow.names if n.startswith('feature_')] == columns
    assert all(str(parquet.schema_arrow.field(n).type) == 'float' for n in columns)
    subset = manifest.loc[manifest.primary].reset_index(drop=True)
    (ROOT / 'cache').mkdir(exist_ok=True)
    array = np.lib.format.open_memmap(ROOT / 'cache/target.npy', mode='w+', dtype=np.float32, shape=(len(subset),2568))
    cursor = 0
    selected_cursor = 0
    for batch in parquet.iter_batches(batch_size=8000, columns=columns):
        values = np.column_stack([batch.column(i).to_numpy() for i in range(2568)])
        assert np.isfinite(values).all()
        keep = manifest.primary.iloc[cursor:cursor+len(values)].to_numpy()
        selected = values[keep]
        array[selected_cursor:selected_cursor+len(selected)] = selected
        cursor += len(values)
        selected_cursor += len(selected)
    assert cursor == 779619 and selected_cursor == len(subset)
    assert np.isin(array[:,739], [0,1]).all()
    subset['is_dll'] = array[:,739].astype(np.int8)
    subset['size_bytes'] = array[:,0].astype(np.int64)
    subset.to_parquet(ROOT / 'data/evaluation_manifest.parquet', index=False)
    array.flush()
    del array
    for name in ['train','test']:
        source_array = np.load(CACHE / f'{name}.npy', mmap_mode='r')
        for first in range(0,len(source_array),20000):
            assert np.isfinite(source_array[first:first+20000]).all()
        source_meta = pd.read_parquet(ROOT / f'data/source_{name}_manifest.parquet')
        source_meta['is_dll'] = source_array[:,739].astype(np.int8)
        source_meta['size_bytes'] = source_array[:,0].astype(np.int64)
        source_meta.to_parquet(ROOT / f'data/evaluation_source_{name}.parquet',index=False)
    params = read(STUDY_A / 'config/baseline_author.json')
    params['device_type'] = 'cpu'
    params['num_threads'] = 48
    save(ROOT / 'config/lightgbm.json', params)
    # Unit-level synthetic checks are separate from the five experimental seeds.
    from run_experiment import threshold
    assert threshold(np.array([.9,.9,.8,.1]), 0.25) > .9
    assert threshold(np.array([.9,.9,.8,.1]), 0.5) == .9
    assert threshold(np.array([.9,.8,.1]), 1/3) == .9
    assert threshold(np.array([.5,.5]), 0.0) > .5
    cfg = {
        'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'authorization':'User delegated scientific completion and implementation on 2026-09-28; no alteration to Studies A-D',
        'status':'FROZEN_BEFORE_NEW_TARGET_PERFORMANCE', 'seeds':SEEDS, 'cpu_only':True, 'threads':48,
        'primary_model':'LightGBM_PREFIX2480', 'secondary_model':'LightGBM_FULL2568_HISTORICAL_SOURCE_FITS',
        'primary_feature_indices':list(range(2480)), 'secondary_feature_indices':list(range(2568)),
        'categorical_indices':[2,3,4,5,6,701,702],
        'representation_decision':'Remove entire 88-column parser-warning group consistently from source and target in primary analysis, prompted by replicated hash-order nondeterminism before performance. Preserve full vectors and use existing full models as sensitivity. This is not the unmodified 2568-column benchmark.',
        'source_release':'joyce8/EMBER2024@9cce319e0a152cd8dfc276b6dea24f7c831df86a',
        'source_training':'Saved 90/10 internal stratified splits of official PE train; each seed has its original fit and validation IDs; fixed 500 rounds, no early stopping, no target tuning',
        'source_test':'First canonical row per SHA from official PE test; original Study A unchanged',
        'target_primary':'Binary operational labels; Win32/Win64/Dot_Net; valid positive first-submission timestamp; exclude union of official source train/test/PE Challenge SHA IDs; natural class distribution',
        'target_sensitivities':['label0_last_analysis_minus_first_submission_ge_30_days','one_lexical_SHA_per_valid_identical_TLSH_after_excluding_source_identical_TLSH','intersection_of_temporal_and_TLSH'],
        'threshold':'For each model and alpha, prediction >= threshold; smallest distinct negative-validation score giving FP <= floor(alpha*n0); if none, nextafter(max_negative,+inf). Ties retained as a group. No target threshold fitting.',
        'source_fpr_targets':[0.01,0.001],
        'metrics':['ROC_AUC','average_precision_sklearn_noninterpolated','TPR','FPR','TP','FP','n_positive','n_negative'],
        'summaries':'All five seeds; mean and sample SD ddof=1; paired differences for fixed conditions. No seed hypothesis tests or bootstrap. Descriptive conditional results.',
        'subgroups':['file_type','is_dll','file_type_by_is_dll','corpus','report_age_0_30_30_365_over365'],
        'sparse_groups':'Always retain counts and rates with denominators; undefined metrics are null; mark strata with fewer than 100 negatives or 100 positives as sparse and avoid general conclusions',
        'input_audit_sha256':sha(ROOT/'qualification/input_audit.json'),
        'warning_diagnostic_sha256':sha(ROOT/'qualification/warning_diagnostic.json'),
        'target_manifest_sha256':sha(ROOT/'data/evaluation_manifest.parquet'),
        'target_cache_sha256':sha(ROOT/'cache/target.npy'),
        'lightgbm_parameters_sha256':sha(ROOT/'config/lightgbm.json'),
        'code_sha256':{p.name:sha(p) for p in (ROOT/'code').glob('*.py')},
        'environment':{'python':platform.python_version(),'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','pyarrow','lightgbm','scikit-learn','py-tlsh']}}
    }
    save(ROOT / 'config/experiment.json', cfg)
    log('experiment_frozen', sha256=sha(ROOT/'config/experiment.json'), target_rows=len(subset))


if __name__ == '__main__':
    main()
