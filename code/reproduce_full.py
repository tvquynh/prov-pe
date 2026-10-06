"""Explicit five-fit reproduction in a NEW directory with supplied source assets.

This is delivered as a recipe. It is not called by default review reproduction.
"""
from pathlib import Path
import argparse
import shutil
import sys


def main(args):
    reference=args.reference_root.resolve(strict=True)
    output=args.output_root.resolve()
    assert not output.exists(), 'Full reproduction requires a new output directory'
    required=['config/experiment.json','config/lightgbm.json','data/evaluation_manifest.parquet',
              'data/evaluation_source_train.parquet','data/evaluation_source_test.parquet',
              'data/historical_near_pair_ledger.parquet','cache/target.npy']
    for rel in required:
        assert (reference/rel).is_file(),f'Full input not provided: {rel}'
    for name in ['train.npy','test.npy']:
        assert (args.source_cache/name).is_file(),name
    for seed in [2026,2027,2028,2029,2030]:
        assert (args.source_package/f'data/internal_split_{seed}.npz').is_file()
        assert (args.source_package/f'models/baseline_seed_{seed}.txt').is_file()
    output.mkdir(parents=True)
    for folder in ['code','config','data','qualification']:
        shutil.copytree(reference/folder,output/folder)
    (output/'cache').mkdir()
    shutil.copy2(reference/'cache/target.npy',output/'cache/target.npy')
    # Change only runtime roots, not the frozen scientific source files.
    sys.path.insert(0,str(output/'code'))
    import common
    common.ROOT=output
    common.STUDY_A=args.source_package.resolve(strict=True)
    common.CACHE=args.source_cache.resolve(strict=True)
    cfg=common.read(output/'config/experiment.json')
    assert common.sha(output/'cache/target.npy')==cfg['target_cache_sha256']
    audit=common.read(reference/'qualification/input_audit.json')['input_identities']
    expected={Path(path).name:details['sha256'] for path,details in audit.items()}
    for name in ['train.npy','test.npy']:
        assert common.sha(common.CACHE/name)==expected[name]
    for seed in common.SEEDS:
        for rel in [f'data/internal_split_{seed}.npz',f'models/baseline_seed_{seed}.txt']:
            assert common.sha(common.STUDY_A/rel)==expected[Path(rel).name]
    import run_experiment
    for seed in common.SEEDS:
        run_experiment.run(seed)
    import analyze_results
    analyze_results.main()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--source-package',type=Path,required=True)
    p.add_argument('--source-cache',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    main(p.parse_args())
