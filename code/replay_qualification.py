"""Regenerate frozen eligibility from existing metadata; no models, PE or VT access.

prepare emits a PRIVATE local input sidecar. replay consumes that sidecar and
compares every frozen row. Neither operation changes the baseline or source.
"""
from pathlib import Path
import argparse
import json
import re
import sys
import time
from cas_revision_common import require, sha, read, save, fresh, environment, inventory, CheckFailed


def canonical(value):
    if not isinstance(value, str) or not re.fullmatch(r'(?:T1)?[0-9A-Fa-f]{70}', value):
        return None
    try:
        obj = tlsh.Tlsh()
        obj.fromTlshStr(value.upper())
        return obj.hexdigest()
    except (ValueError, TypeError):
        return None


def prepare(args, output, report):
    config = read(args.source_config)
    inputs = {}

    def source(key):
        item = config[key]
        p = Path(item['path']).resolve(strict=True)
        require(not output.is_relative_to(p.parent), 'OUTPUT_UNDER_SOURCE_DIRECTORY')
        actual = sha(p)
        require(actual == item['sha256'], 'INPUT_IDENTITY_MISMATCH: ' + key)
        inputs[key] = {'path': str(p), 'sha256': actual, 'bytes': p.stat().st_size}
        print('Verified input:', key, flush=True)
        return p

    feature_path = source('target_features')
    target = pd.read_parquet(feature_path, columns=['sha256', 'tlsh', 'file_type', 'label', 'first_submission_date', 'last_analysis_date'])
    target['target_row'] = np.arange(len(target), dtype=np.int64)
    target = target.rename(columns={'label': 'recorded_label', 'tlsh': 'raw_tlsh'})
    provenance = pd.read_parquet(source('provenance'), columns=['sha256', 'dataset_row', 'corpus', 'vt_status', 'malicious_count', 'label_reason', 'response_received_utc'])
    require(target.sha256.is_unique and provenance.sha256.is_unique, 'TARGET_DUPLICATE_SHA')
    target = target.merge(provenance, on='sha256', how='left', sort=False, validate='one_to_one')
    require(np.array_equal(target.target_row, target.dataset_row), 'TARGET_ROW_LINKAGE')
    require(target.vt_status.notna().all(), 'MISSING_REPORT_STATUS')
    target.to_parquet(output / 'target_inputs.parquet', index=False)
    for split in ['train', 'test', 'challenge']:
        manifest = pd.read_parquet(source(split + '_membership'))
        meta = pd.read_parquet(source(split + '_metadata'), columns=['sha256', 'tlsh'])
        selected = meta.iloc[manifest.source_row].reset_index(drop=True)
        require(np.array_equal(selected.sha256, manifest.sha256), 'SOURCE_ROW_LINKAGE: ' + split)
        require(manifest.file_type.isin(['Win32', 'Win64', 'Dot_Net']).all(), 'NON_PE_SOURCE_GROUP')
        manifest['raw_tlsh'] = selected.tlsh
        manifest['canonical_row'] = np.arange(len(manifest), dtype=np.int64)
        manifest.to_parquet(output / ('source_' + split + '.parquet'), index=False)
        print('Prepared source metadata:', split, len(manifest), flush=True)
    historical = pd.read_parquet(source('historical_candidates'), columns=['sha256', 'collection_snapshot', 'ember_neardup', 'ember_neardup_min_distance', 'ember_neardup_reference_sha256', 'internal_neardup'])
    require(historical.sha256.is_unique, 'HISTORICAL_DUPLICATE_SHA')
    historical.to_parquet(output / 'historical_candidates.parquet', index=False)
    save(output / 'SOURCE_IDENTITIES.json', inputs)
    schemas = {}
    for p in output.glob('*.parquet'):
        table = pd.read_parquet(p)
        schemas[p.name] = {'rows': len(table), 'columns': {name: {'dtype': str(table[name].dtype), 'missing': int(table[name].isna().sum())} for name in table}}
    save(output / 'SCHEMA.json', {'files': schemas, 'timestamps': 'first_submission_date and last_analysis_date: UTC Unix seconds; response_received_utc: UTC ISO 8601; intervals: seconds / 86400',
                                'recorded_label_role': 'Comparison only; operational label is regenerated from vt_status and malicious_count.',
                                'source_canonical_order': 'Exact pinned membership row order; select metadata by source_row and verify SHA. Repeated test SHA uses first canonical row.',
                                'missingness': 'Preserve nulls; not_found produces unresolved label, no synthetic timestamp is inserted.',
                                'distribution_status': 'PRIVATE_LOCAL_STAGING; redistribution and external access unconfirmed'})
    save(output / 'INPUT_MANIFEST.json', {'files': inventory(output), 'status': 'PRIVATE_LOCAL_INPUTS_PREPARED_NOT_RELEASED'})
    report.update(status='PASS', input_files=inputs, sidecar_files=read(output / 'INPUT_MANIFEST.json')['files'], source_tensor_arrays_read=0,
                  scope='Metadata columns and file hashes only; no feature-vector loading, PE parsing or database modification.')


def replay(args, output, report):
    sidecar = args.sidecar_root.resolve(strict=True)
    for name, item in read(sidecar / 'INPUT_MANIFEST.json')['files'].items():
        require(sha(sidecar / name) == item['sha256'], 'SIDECAR_IDENTITY_MISMATCH: ' + name)
    base = args.root.resolve(strict=True)
    target = pd.read_parquet(sidecar / 'target_inputs.parquet')
    expected = pd.read_parquet(base / 'review_data/inventory_eligibility.parquet')
    evaluated = pd.read_parquet(base / 'review_data/target.parquet')
    require(target.sha256.is_unique and np.array_equal(target.sha256, expected.sha256), 'INVENTORY_SHA_ORDER')
    require(np.array_equal(target.target_row, np.arange(len(target))), 'INVENTORY_ROW_ID')
    target['label'] = np.where(target.vt_status.eq('not_found'), -1, np.where(target.malicious_count.eq(0), 0, np.where(target.malicious_count.ge(5), 1, -1)))
    require(np.array_equal(target.label, target.recorded_label) and np.array_equal(target.label, expected.label), 'OPERATIONAL_LABEL_DISCREPANCY')
    require(target.vt_status.isin(['ok', 'not_found']).all(), 'UNEXPECTED_REPORT_STATUS')
    require(target.loc[target.vt_status.eq('ok'), 'malicious_count'].notna().all(), 'MISSING_USABLE_MALICIOUS_COUNT')
    target['tlsh_canonical'] = target.raw_tlsh.map(canonical)
    target['tlsh_assessable'] = target.tlsh_canonical.notna()
    source_frames = []
    overlaps = {}
    for split in ['train', 'test', 'challenge']:
        source = pd.read_parquet(sidecar / ('source_' + split + '.parquet'))
        require(np.array_equal(source.canonical_row, np.arange(len(source))), 'SOURCE_CANONICAL_ORDER')
        source['tlsh_canonical'] = source.raw_tlsh.map(canonical)
        target['overlap_' + split] = target.sha256.isin(set(source.sha256))
        overlaps[split] = int(target['overlap_' + split].sum())
        if split == 'test':
            source_test = pd.read_parquet(base / 'review_data/source_test.parquet')
            canonical_test = source.drop_duplicates('sha256', keep='first')
            require(np.array_equal(canonical_test.sha256, source_test.sha256), 'CANONICAL_TEST_SHA_ORDER')
            require(np.array_equal(canonical_test.label, source_test.label), 'CANONICAL_TEST_LABEL')
        source_frames.append(source[['sha256', 'tlsh_canonical']])
        print('Replayed membership:', split, flush=True)
    source_all = pd.concat(source_frames, ignore_index=True)
    source_tls = set(source_all.tlsh_canonical.dropna())
    target['overlap_source_union'] = target[['overlap_train', 'overlap_test', 'overlap_challenge']].any(axis=1)
    target['overlap_source_identical_tlsh'] = target.tlsh_canonical.isin(source_tls)
    target['eligible_type'] = target.file_type.isin(['Win32', 'Win64', 'Dot_Net'])
    target['eligible_label'] = target.label.isin([0, 1])
    target['eligible_time'] = target.first_submission_date.notna() & target.first_submission_date.gt(0)
    target['primary'] = target.eligible_type & target.eligible_label & target.eligible_time & ~target.overlap_source_union
    window = (target.last_analysis_date - target.first_submission_date) / 86400.0
    target['temporal_sensitivity'] = target.primary & (target.label.eq(1) | (window.notna() & window.ge(30)))
    age = (pd.to_datetime(target.response_received_utc, utc=True) - pd.to_datetime(target.last_analysis_date, unit='s', utc=True)).dt.total_seconds() / 86400.0
    choices = target.loc[target.primary & target.tlsh_assessable & ~target.overlap_source_identical_tlsh].sort_values('sha256')
    representatives = choices.drop_duplicates('tlsh_canonical', keep='first').sha256
    target['identical_tlsh_sensitivity'] = target.sha256.isin(set(representatives))
    target['temporal_and_tlsh_sensitivity'] = target.temporal_sensitivity & target.identical_tlsh_sensitivity
    masks = ['eligible_type', 'eligible_label', 'eligible_time', 'overlap_source_union', 'primary', 'temporal_sensitivity', 'identical_tlsh_sensitivity', 'temporal_and_tlsh_sensitivity']
    disagreements = {key: int(np.count_nonzero(target[key].to_numpy() != expected[key].to_numpy())) for key in masks}
    require(not any(disagreements.values()), 'FROZEN_MASK_DISCREPANCY: ' + str(disagreements))
    selected = target.loc[target.primary]
    require(np.array_equal(selected.sha256, evaluated.sha256) and np.array_equal(selected.target_row, evaluated.target_row), 'SCORE_ROW_LINKAGE')
    require(np.allclose(age[target.primary], evaluated.report_age_days, atol=1e-12, rtol=0, equal_nan=True), 'REPORT_AGE_DISCREPANCY')
    require(source_all.groupby('sha256').tlsh_canonical.nunique().max() == 1, 'SOURCE_DIGEST_CONFLICT')
    source_unique = source_all.drop_duplicates('sha256').rename(columns={'sha256': 'ember_neardup_reference_sha256', 'tlsh_canonical': 'source_tlsh'})
    historical = pd.read_parquet(sidecar / 'historical_candidates.parquet')
    pairs = target[['sha256', 'tlsh_canonical', 'primary', 'label']].merge(historical, on='sha256', how='left', sort=False, validate='one_to_one').merge(source_unique, on='ember_neardup_reference_sha256', how='left', sort=False, validate='many_to_one')
    require(np.array_equal(pairs.sha256, target.sha256), 'NAMED_PAIR_ROW_ORDER')
    pairs['verified_distance'] = [tlsh.diff(a, b) if isinstance(a, str) and isinstance(b, str) else np.nan for a, b in zip(pairs.tlsh_canonical, pairs.source_tlsh)]
    pairs['verified_le30'] = pairs.verified_distance.le(30)
    target['known_near_control'] = target.primary & ~pairs.verified_le30
    require(np.array_equal(target.loc[target.primary, 'known_near_control'], evaluated.known_near_control), 'KNOWN_NEIGHBOR_MASK_DISCREPANCY')
    views = ['primary', 'temporal_sensitivity', 'identical_tlsh_sensitivity', 'temporal_and_tlsh_sensitivity', 'known_near_control']
    counts = {view: {'rows': int(target[view].sum()), 'label_counts': {str(int(k)): int(v) for k, v in target.loc[target[view], 'label'].value_counts().items()}} for view in views}
    old = read(base / 'qualification/input_audit.json')
    require(overlaps == old['overlap_counts'], 'OVERLAP_COUNTS')
    require(int(target.overlap_source_union.sum()) == old['source_union_overlap'], 'UNION_COUNT')
    for view in views[:-1]:
        require(counts[view] == old['cohorts'][view], 'CLASS_COUNTS: ' + view)
    prior_near = read(base / 'qualification/historical_near_pair_audit.json')
    require(counts['known_near_control']['label_counts'] == prior_near['retained_primary_label_counts'], 'NEIGHBOR_CLASS_COUNTS')
    sequence = []
    remaining = np.ones(len(target), dtype=bool)
    for reason, keep in [('binary_label', target.eligible_label), ('three_PE_groups', target.eligible_type), ('first_submission_available', target.eligible_time), ('no_official_SHA_overlap', ~target.overlap_source_union)]:
        before = int(remaining.sum()); remaining &= keep.to_numpy()
        sequence.append({'condition': reason, 'before': before, 'excluded': before - int(remaining.sum()), 'after': int(remaining.sum())})
    require(sequence == old['sequential_exclusions'], 'SEQUENTIAL_EXCLUSION_COUNTS')
    target[['sha256', 'target_row', 'label', *masks, 'known_near_control']].to_parquet(output / 'reconstructed_masks.parquet', index=False)
    pairs.to_parquet(output / 'reconstructed_named_pairs.parquet', index=False)
    report.update(status='PASS', inventory_rows=len(target), masks_compared=disagreements, known_neighbor_mask_disagreements=0,
                  label_disagreements=0, source_test_order_and_labels_exact=True, overlaps=overlaps,
                  source_union_overlap=int(target.overlap_source_union.sum()), cohorts=counts, sequential_exclusions=sequence,
                  report_age_row_comparison='PASS_atol_1e-12_days_rtol_0',
                  unresolved_reasons={str(k): int(v) for k, v in target.loc[target.label.eq(-1), 'label_reason'].value_counts(dropna=False).items()},
                  named_pair_scope={'reference_in_pinned_source': int(pairs.source_tlsh.notna().sum()), 'verified_le30': int(pairs.verified_le30.sum()), 'excluded_primary': int((pairs.primary & pairs.verified_le30).sum()), 'exhaustive_search': False},
                  input_manifest_sha256=sha(sidecar / 'INPUT_MANIFEST.json'), baseline_identity_sha256=sha(base / 'review_data/identities.json'),
                  outputs=inventory(output), distribution_status='PRIVATE_LOCAL_AUDIT; inputs require rights and an actual external access route')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'replay'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-config', type=Path)
    parser.add_argument('--sidecar-root', type=Path)
    args = parser.parse_args()
    protected = [args.root] + ([args.sidecar_root] if args.sidecar_root else [])
    out = fresh(args.output, protected)
    result = {'status': 'STARTED', 'operation': args.operation, 'environment': environment(['numpy', 'pandas', 'pyarrow', 'py-tlsh']),
              'training_runs': 0, 'model_inference_calls': 0, 'VT_requests': 0, 'PE_parses': 0, 'new_experiments': 0}
    start = time.perf_counter()
    exit_code = 0
    try:
        import numpy as np
        import pandas as pd
        import pyarrow
        import tlsh
        require(args.source_config is not None if args.operation == 'prepare' else args.sidecar_root is not None, 'MISSING_OPERATION_INPUT')
        (prepare if args.operation == 'prepare' else replay)(args, out, result)
    except ImportError as exc:
        result.update(status='ENVIRONMENT_BLOCKED', error=str(exc)); exit_code = 2
    except Exception as exc:
        result.update(status='CHECK_FAILED', error=repr(exc)); exit_code = 1
    result['elapsed_seconds'] = time.perf_counter() - start
    save(out / 'receipt.json', result)
    print(json.dumps({k: v for k, v in result.items() if k in ['status', 'error', 'elapsed_seconds', 'cohorts', 'overlaps', 'masks_compared']}, indent=2), flush=True)
    sys.exit(exit_code)
