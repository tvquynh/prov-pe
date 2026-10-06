"""Outcome-blind source membership, lineage, and cohort materialization."""
import os
os.environ['OMP_NUM_THREADS'] = '16'
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tlsh
from common import ROOT, DATASET, STUDY_A, SOURCE, CACHE, SEEDS, sha, read, save, log


def canonical_tlsh(value):
    if not isinstance(value, str) or not re.fullmatch(r'(?:T1)?[0-9A-Fa-f]{70}', value):
        return None
    try:
        obj = tlsh.Tlsh()
        obj.fromTlshStr(value.upper())
        return obj.hexdigest()
    except (ValueError, TypeError):
        return None


def main():
    folder = ROOT / 'data'
    folder.mkdir(parents=True, exist_ok=True)
    report = ROOT / 'qualification/input_audit.json'
    assert not report.exists(), 'Do not overwrite a completed audit'
    identities = {}
    def verify(path, expected=None):
        value = sha(path)
        assert expected is None or value == expected, ('HASH_MISMATCH', str(path), value, expected)
        identities[str(path)] = {'sha256': value, 'bytes': path.stat().st_size, 'expected_sha256': expected}
        log('verified', file=str(path), sha256=value)
    validation = read(DATASET / 'parquet_vt_20260927/VALIDATION_REPORT.json')
    target_path = DATASET / 'parquet_vt_20260927/collected_pe_ember_v3.parquet'
    provenance_path = DATASET / 'parquet_vt_20260927/sample_provenance.parquet'
    verify(target_path, validation['dataset_sha256'])
    verify(provenance_path, validation['provenance_sha256'])
    fields = ['sha256','md5','tlsh','file_type','label','first_submission_date','last_analysis_date']
    target = pd.read_parquet(target_path, columns=fields)
    target['target_row'] = np.arange(len(target))
    provenance = pd.read_parquet(provenance_path)
    assert len(target) == 779619 and target.sha256.is_unique and provenance.sha256.is_unique
    assert target.sha256.notna().all() and target.sha256.str.fullmatch('[0-9a-f]{64}').all()
    target = target.merge(provenance.drop(columns=['label']), on='sha256', validate='one_to_one', sort=False)
    assert np.array_equal(target.target_row, np.arange(len(target)))
    assert np.array_equal(target.target_row, target.dataset_row)
    label_expected = np.where(target.vt_status.eq('not_found'), -1,
                              np.where(target.malicious_count.eq(0), 0,
                                       np.where(target.malicious_count.ge(5), 1, -1)))
    assert np.array_equal(target.label, label_expected)
    target['tlsh_canonical'] = target.tlsh.map(canonical_tlsh)
    target['tlsh_assessable'] = target.tlsh_canonical.notna()
    source_cfg = read(STUDY_A / 'config/experiment.json')
    checksum_map = {}
    for line in (STUDY_A/'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        checksum_map[name.strip().lstrip('*').replace('\\','/')] = digest
    source = {}
    overlaps = {}
    source_tls = set()
    for name in ['train','test','challenge']:
        manifest_path = STUDY_A / f'data/{name}_manifest.parquet'
        verify(manifest_path, checksum_map.get(f'data/{name}_manifest.parquet'))
        manifest = pd.read_parquet(manifest_path)
        parquet = SOURCE / f'ember2024_{name}.parquet'
        verify(parquet, source_cfg['source_hashes'][name])
        meta = pd.read_parquet(parquet, columns=['sha256','tlsh','first_submission_date','last_analysis_date'])
        selected = meta.iloc[manifest.source_row].reset_index(drop=True)
        assert np.array_equal(selected.sha256, manifest.sha256)
        manifest['tlsh_canonical'] = selected.tlsh.map(canonical_tlsh)
        manifest['first_submission_date'] = selected.first_submission_date
        manifest['last_analysis_date'] = selected.last_analysis_date
        manifest['canonical_row'] = np.arange(len(manifest))
        manifest['unique_sha_representative'] = ~manifest.sha256.duplicated(keep='first')
        source_tls.update(manifest.tlsh_canonical.dropna())
        manifest.to_parquet(folder / f'source_{name}_manifest.parquet', index=False)
        source[name] = manifest
        mask = target.sha256.isin(set(manifest.sha256))
        target[f'overlap_{name}'] = mask
        overlaps[name] = int(mask.sum())
        log('membership_checked', split=name, rows=len(manifest), unique_sha=int(manifest.sha256.nunique()), target_overlap=int(mask.sum()))
    for seed in SEEDS:
        for rel in [f'models/baseline_seed_{seed}.txt', f'data/internal_split_{seed}.npz']:
            verify(STUDY_A / rel, checksum_map[rel])
        split = np.load(STUDY_A / f'data/internal_split_{seed}.npz')
        assert len(split['fit']) == 2106000 and len(split['validation']) == 234000
        joined = np.concatenate([split['fit'], split['validation']])
        assert np.array_equal(np.sort(joined), np.arange(len(source['train'])))
        for role in ['fit','validation']:
            ids = source['train'].iloc[split[role]].sha256
            target[f'overlap_{role}_{seed}'] = target.sha256.isin(set(ids))
    target['overlap_source_union'] = target[['overlap_train','overlap_test','overlap_challenge']].any(axis=1)
    target['overlap_source_identical_tlsh'] = target.tlsh_canonical.isin(source_tls)
    target['eligible_type'] = target.file_type.isin(['Win32','Win64','Dot_Net'])
    target['eligible_label'] = target.label.isin([0,1])
    target['eligible_time'] = target.first_submission_date.notna() & target.first_submission_date.gt(0)
    target['primary'] = target.eligible_type & target.eligible_label & target.eligible_time & ~target.overlap_source_union
    window = (target.last_analysis_date - target.first_submission_date) / 86400.0
    target['first_to_analysis_days'] = window
    target['temporal_sensitivity'] = target.primary & (target.label.eq(1) | (window.notna() & window.ge(30)))
    target['report_age_days'] = (pd.to_datetime(target.response_received_utc, utc=True) - pd.to_datetime(target.last_analysis_date, unit='s', utc=True)).dt.total_seconds()/86400
    # Representatives depend only on a verified digest and SHA lexical order, never on score or label.
    candidate = target.loc[target.primary & target.tlsh_assessable & ~target.overlap_source_identical_tlsh].sort_values('sha256')
    representatives = candidate.drop_duplicates('tlsh_canonical', keep='first').sha256
    target['identical_tlsh_sensitivity'] = target.sha256.isin(set(representatives))
    target['temporal_and_tlsh_sensitivity'] = target.temporal_sensitivity & target.identical_tlsh_sensitivity
    target.to_parquet(folder / 'target_manifest.parquet', index=False)
    groups = target.loc[target.tlsh_assessable].groupby('tlsh_canonical').agg(samples=('sha256','size'), n_labels=('label','nunique'), n_sources=('corpus','nunique'))
    groups.to_parquet(folder / 'identical_tlsh_groups.parquet')
    counts = {}
    for cohort in ['primary','temporal_sensitivity','identical_tlsh_sensitivity','temporal_and_tlsh_sensitivity']:
        frame = target.loc[target[cohort]]
        counts[cohort] = {'rows': len(frame), 'label_counts': {str(k): int(v) for k,v in frame.label.value_counts().items()}}
        frame.groupby(['corpus','file_type','label']).size().rename('samples').reset_index().to_csv(folder / f'{cohort}_composition.csv',index=False)
    sequence = []
    remaining = np.ones(len(target), dtype=bool)
    for reason, keep in [('binary_label',target.eligible_label),('three_PE_groups',target.eligible_type),('first_submission_available',target.eligible_time),('no_official_SHA_overlap',~target.overlap_source_union)]:
        before = int(remaining.sum())
        remaining &= keep.to_numpy()
        sequence.append({'condition':reason,'before':before,'excluded':before-int(remaining.sum()),'after':int(remaining.sum())})
    assert np.array_equal(remaining, target.primary)
    cache_identity = read(STUDY_A / 'logs/canonical_feature_cache_identity.json')['arrays']
    for name in ['train','test','challenge']:
        verify(CACHE / f'{name}.npy', cache_identity[name]['sha256'])
    db = DATASET / 'snapshots/COLLECTED_PE_779619_20260923/COLLECTED_PE_779619_20260923.db'
    verify(db, '2f8f86e94057cab6c561ce08ed0daf9cab5e4a71ec7ba703a3ca5e53d5508885')
    save(report, {'status':'PASS','outcome_blind':True,'input_identities':identities,'overlap_counts':overlaps,
                  'source_union_overlap':int(target.overlap_source_union.sum()),'cohorts':counts,'sequential_exclusions':sequence,
                  'tls':{'assessable':int(target.tlsh_assessable.sum()),'unassessable':int((~target.tlsh_assessable).sum()),
                         'duplicate_digest_groups':int(groups.samples.gt(1).sum()),'records_in_duplicate_digest_groups':int(groups.loc[groups.samples.gt(1),'samples'].sum()),
                         'mixed_label_groups':int((groups.n_labels.gt(1)&groups.samples.gt(1)).sum()),
                         'source_digest_overlap':int(target.overlap_source_identical_tlsh.sum()),'scope':'Exact valid TLSH equality only; not a complete near-duplicate or family-independence guarantee'},
                  'target_manifest_sha256':sha(folder/'target_manifest.parquet'), 'no_model_scores_computed':True})
    log('audit_complete', cohorts=counts, sequential_exclusions=sequence)


if __name__ == '__main__':
    main()
