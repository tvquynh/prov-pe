"""Optional full-input replay recipe. This is NOT called by compact verification.

Supply a JSON object mapping frozen input IDs to local paths. This wrapper
checks every input hash, refuses an existing output, and executes the archived
training functions without changing their scientific parameters. It will fit
five models for the explicitly chosen role. Full matrices are not in the ZIP.
"""
import argparse,shutil
from pathlib import Path
from datetime import datetime,timezone
from revision_common import ROOT,read,save,sha,SEEDS

def main(args):
    output=Path(args.output).resolve()
    if output.exists():raise RuntimeError('Choose a new output directory')
    cfg=read(ROOT/'config/revision_analysis.json');mapping=read(args.input_map)
    missing=set(cfg['inputs'])-set(mapping)
    if missing:raise ValueError('Missing frozen input paths: '+', '.join(sorted(missing)))
    for key,identity in cfg['inputs'].items():
        p=Path(mapping[key]).resolve(strict=True)
        if sha(p)!=identity['sha256']:raise RuntimeError('Input identity mismatch: '+key)
        identity['path']=str(p)
    cfg['reproduction_of_config_sha256']=sha(ROOT/'config/revision_analysis.json')
    cfg['reproduction_started_utc']=datetime.now(timezone.utc).isoformat()
    for name in ['config','code','data','models','predictions','results','logs','cache']:(output/name).mkdir(parents=True,exist_ok=False)
    for name in ['train_revision.py','revision_common.py']:shutil.copy2(ROOT/'code'/name,output/'code'/name)
    shutil.copy2(ROOT/'config/training_code_hashes.json',output/'config/training_code_hashes.json')
    for seed in SEEDS:shutil.copy2(ROOT/f'data/source_diagnostic_split_{seed}.npy',output/f'data/source_diagnostic_split_{seed}.npy')
    save(output/'config/revision_analysis.json',cfg)
    import train_revision as training
    training.ROOT=output;training.WORK=output;training.ORIGINAL=ROOT.parent
    for seed in SEEDS:
        (training.source if args.role=='source' else training.linear)(seed,cfg)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--role',choices=['source','linear'],required=True);p.add_argument('--input-map',required=True);p.add_argument('--output',required=True);main(p.parse_args())
