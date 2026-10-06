"""Assemble and verify one review ZIP after scientific execution and visual QA."""
import json
import re
import shutil
import subprocess
import zipfile
import fitz
from common import *


def copy(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)


def main():
    bundle=ROOT/'review_package'
    assert read(ROOT/'qualification/reproduction.json')['status']=='PASS'
    portable=read(bundle/'reproduced/review_reproduction.json')
    assert portable['status']=='PASS' and portable['training_runs']==0
    save(ROOT/'qualification/portable_review_reproduction.json',portable)
    correction=read(ROOT/'qualification/analysis_technical_correction.json')
    assert all(sha(ROOT/'results'/name)==digest for name,digest in correction['partial_csv_sha256'].items())
    correction['post_fix_partial_csv_byte_identity_verified']=True
    save(ROOT/'qualification/analysis_technical_correction.json',correction)
    for folder in ['code','config','qualification','results','logs','technical_archive','environment','descriptive_inputs']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                copy(p,bundle/p.relative_to(ROOT))
    for p in (ROOT/'paper').rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.tex','.bib','.bbl','.cls','.sty','.pdf','.png','.jpeg'}:
            copy(p,bundle/p.relative_to(ROOT))
    for p in list(ROOT.glob('*.md'))+list(ROOT.glob('*.txt'))+list(ROOT.glob('*.ps1')):
        if p.name!='WORK_STATE.md':copy(p,bundle/p.name)
    for p in (ROOT.parent/'received_review_20260928').glob('*'):
        if p.is_file():copy(p,bundle/'prior_review'/p.name)
    copy(ROOT.parent/'JISA_COLLECTED_PE_MANUSCRIPT_DRAFT_20260928.pdf',bundle/'prior_review/original_draft.pdf')
    copy(ROOT.parent/'reference_template/JISA_GUIDE_SUPPLIED_20260928.txt',bundle/'journal_reference/JISA_GUIDE_SUPPLIED_20260928.txt')
    # Check figure and table regeneration using only inputs enclosed in the ZIP.
    names=[p.name for p in (ROOT/'paper/generated').glob('*.tex') if p.name not in ['declarations.tex','collection_history.tex','reproduction_statement.tex']]
    subprocess.run([str(Path(__import__('sys').executable)),'-X','utf8',str(bundle/'code/build_paper_assets.py')],check=True)
    for name in names:assert sha(bundle/'paper/generated'/name)==sha(ROOT/'paper/generated'/name),name
    for p in (ROOT/'paper/figures').glob('*.png'):
        assert sha(bundle/'paper/figures'/p.name)==sha(p),p.name
    save(bundle/'qualification/portable_assets_reproduction.json',{'status':'PASS','numerical_tex_byte_identity':names,
        'png_byte_identity':[p.name for p in (ROOT/'paper/figures').glob('*.png')],
        'scope':'Regenerated from enclosed result CSVs, audit JSON and minimal target labels. PDF metadata timestamps need not match.'})
    resource=ROOT/'feature_resource_candidate'
    for p in (ROOT/'config').glob('*.json'):copy(p,resource/'config'/p.name)
    (resource/'README.md').write_text('# Local full feature candidate\n\nThis directory contains the byte-preserved 2,568-column collected feature Parquet, sample provenance, complete target eligibility and historical-neighbor ledgers. The primary reference uses columns 0 through 2479.\n\nThis is a local candidate prepared for the owner, not a completed public deposit or license grant. No original PE binaries, raw VT responses or credentials are included. Confirm the permitted distribution scope and durable access route before public release. Full source-training assets are separate and identified in the review input audit.\n',encoding='utf8')
    save(resource/'RESOURCE_STATUS.json',{'status':'LOCAL_FEATURE_CANDIDATE_PREPARED','public_deposit':False,'public_license':None,
        'files':{str(p.relative_to(resource)).replace('\\','/'):sha(p) for p in resource.rglob('*') if p.is_file() and p.name!='RESOURCE_STATUS.json'}})
    save(bundle/'FEATURE_RESOURCE_POINTER.json',{'local_path':str(resource),'status':'LOCAL_CANDIDATE_NOT_PUBLIC_DEPOSIT',
        'resource_status_sha256':sha(resource/'RESOURCE_STATUS.json'),'resource_files':read(resource/'RESOURCE_STATUS.json')['files']})
    preserved=ROOT.parent/'JISA_COLLECTED_PE_MANUSCRIPT_DRAFT_20260928.zip'
    assert sha(preserved)=='3a6daa6f0b972e7a29145d05a6045124d18a68b55310b98d713e970f6c47376c'
    pdfs={}
    for name in ['main','supplement']:
        with fitz.open(bundle/'paper'/(name+'.pdf')) as doc:
            assert not any('??' in page.get_text() for page in doc)
            pdfs[name]={'pages':len(doc),'sha256':sha(bundle/'paper'/(name+'.pdf'))}
    report={'technical_status':'PASS','scientific_execution':'COMPLETE','new_primary_cpu_fits':5,
        'training_seeds':SEEDS,'historical_source_models_scored':5,'source_training_only':True,
        'external_primary_rows':742273,'per_seed_main_rows':120,'per_seed_subgroup_rows':340,
        'figures':4,'main_tables':5,'pdfs':pdfs,'original_draft_zip_unchanged':True,
        'new_VT_requests':0,'new_bootstrap_runs':0,'thesis_studies_modified':False,
        'journal_submission_status':'AUTHOR_FACTS_AND_DURABLE_DATA_ACCESS_PENDING',
        'outstanding':['author-confirmed funding, interests, CRediT and final approval','permitted redistribution and durable access route','any recoverable original benign collection selection details'],
        'human_author_approval_asserted':False,'actual_journal_submission':False}
    save(bundle/'DELIVERY_STATUS.json',report)
    files=sorted(p for p in bundle.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt' and '__pycache__' not in p.parts)
    # Keep an original prior-review checksum file intact but also cover it in the outer manifest.
    files += [bundle/'prior_review/SHA256SUMS.txt'] if (bundle/'prior_review/SHA256SUMS.txt').exists() else []
    files=sorted(set(files))
    lines=[sha(p)+'  '+str(p.relative_to(bundle)).replace('\\','/') for p in files]
    (bundle/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    output=ROOT.parent/'JISA_COLLECTED_PE_REVISED_REVIEW_PACKAGE_20260928.zip'
    assert not output.exists(),'Do not silently overwrite a delivered archive'
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in files+[bundle/'SHA256SUMS.txt']:
            z.write(p,str(p.relative_to(bundle)).replace('\\','/'))
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for line in lines:
            digest,rel=line.split('  ',1)
            assert __import__('hashlib').sha256(z.read(rel)).hexdigest()==digest,rel
    receipt={'status':'PASS','zip':str(output),'bytes':output.stat().st_size,'sha256':sha(output),'files':len(files)+1,
        'archive_member_sha256_verified':True,'submission_status':report['journal_submission_status']}
    save(ROOT/'DELIVERY_RECEIPT.json',receipt)
    text=json.dumps(receipt,indent=2)
    print(text,flush=True)


if __name__=='__main__':main()
