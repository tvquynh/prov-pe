"""Create the approved unbalanced, Sunday-based UTC temporal release."""
from pathlib import Path
import argparse,hashlib,json,time,sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE=Path(__file__).resolve().parents[1]
SRC=BASE/'parquet_vt_20260927'
pa.set_cpu_count(8)

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def csv(out,name,x):x.to_csv(out/name,index=False,encoding='utf-8-sig')

def main(out,resume=False):
    start=time.monotonic()
    if out.exists() and not resume:raise FileExistsError('Choose a new output directory')
    v=json.loads((SRC/'VALIDATION_REPORT.json').read_text())
    assert v['status']=='PASS'
    assert digest(SRC/'collected_pe_ember_v3.parquet')==v['dataset_sha256']
    assert digest(SRC/'sample_provenance.parquet')==v['provenance_sha256']
    out.mkdir(exist_ok=resume);(out/'figures').mkdir(exist_ok=resume)
    dump(out/'BUILD_STATE.json',{'status':'BUILDING'})
    t=pq.read_table(SRC/'collected_pe_ember_v3.parquet',columns=['sha256','label','file_type','first_submission_date'] if resume else None)
    df=t.select(['sha256','label','file_type','first_submission_date']).to_pandas()
    pr=pq.read_table(SRC/'sample_provenance.parquet').to_pandas()
    assert df.sha256.is_unique and pr.sha256.is_unique
    assert df.sha256.tolist()==pr.sha256.tolist()
    df['source_row']=np.arange(len(df))
    date=pd.to_datetime(df.first_submission_date,unit='s',utc=True,errors='coerce')
    received=pd.to_datetime(pr.response_received_utc,utc=True,errors='raise')
    binary=df.label.isin([0,1]);valid=date.notna() & (df.first_submission_date>0) & (date<=received)
    keep=binary & valid
    excluded=df.loc[~keep].copy()
    excluded['exclusion_reason']=np.where(~binary[~keep],'LABEL_MINUS1',np.where(date[~keep].isna(),'MISSING_FIRST_SUBMISSION_DATE','INVALID_FIRST_SUBMISSION_DATE'))
    excluded=excluded.merge(pr[['sha256','label_reason','vt_status']],on='sha256',validate='one_to_one')
    csv(out,'excluded_samples.csv',excluded)
    selected=df.loc[keep].copy();selected['first_submission_utc']=date[keep].astype('datetime64[ms, UTC]')
    selected['first_submission_date']=selected.first_submission_date.astype('int64')
    selected.sort_values(['first_submission_utc','sha256'],inplace=True)
    origin=selected.first_submission_utc.min().normalize()
    origin-=pd.Timedelta(days=(origin.dayofweek+1)%7)
    selected['week_id']=((selected.first_submission_utc-origin).dt.total_seconds()//604800).astype('int64')
    selected['week_start_utc']=origin+pd.to_timedelta(selected.week_id*7,unit='D')
    selected['week_end_exclusive_utc']=selected.week_start_utc+pd.Timedelta(days=7)
    selected['label_output_row']=selected.groupby('label').cumcount()
    assert len(selected)==749512 and selected.label.value_counts().to_dict()=={1:553322,0:196190}
    assert selected.week_id.min()==0 and origin.dayofweek==6
    assert (selected.first_submission_utc>=selected.week_start_utc).all()
    assert (selected.first_submission_utc<selected.week_end_exclusive_utc).all()
    assert len(selected)+len(excluded)==len(df)
    # Boundary tests: Sunday start belongs to the new week, prior second does not.
    assert int((origin+pd.Timedelta(days=7)-origin).total_seconds()//604800)==1
    assert int((origin+pd.Timedelta(days=7,seconds=-1)-origin).total_seconds()//604800)==0
    outputs={}
    if resume:
        receipt=json.loads((out/'FINALIZATION_RECOVERY.json').read_text())
        assert receipt['full_label_equality_verified_before_error'] is True
    for label in [0,1]:
        s=selected[selected.label==label]
        positions=s.source_row.to_numpy();weeks=s.week_id.to_numpy()
        dest=out/f'label_{label}.parquet'
        if resume:
            previous=receipt['outputs'][str(label)]
            assert digest(dest)==previous['sha256']
            actual=pq.read_table(dest,columns=['sha256','label','first_submission_date','week_id']).to_pandas()
            assert actual.sha256.tolist()==s.sha256.tolist()
            assert actual.week_id.tolist()==s.week_id.tolist()
            assert actual.label.eq(label).all() and len(actual)==previous['rows']
            outputs[str(label)]=previous
            print('Previously verified label file unchanged',label,flush=True)
            continue
        with pq.ParquetWriter(dest,t.schema,compression='zstd',compression_level=3) as writer:
            for offset in range(0,len(s),8192):
                part=t.take(pa.array(positions[offset:offset+8192]))
                part=part.set_column(t.schema.get_field_index('week_id'),t.schema.field('week_id'),pa.array(weeks[offset:offset+8192],type=pa.int64()))
                writer.write_table(part,row_group_size=part.num_rows)
                if offset%65536==0:print('Writing label',label,'rows',offset,flush=True)
        read=pq.ParquetFile(dest);offset=0
        for batch in read.iter_batches(batch_size=8192):
            actual=pa.Table.from_batches([batch])
            expected=t.take(pa.array(positions[offset:offset+len(batch)]))
            expected=expected.set_column(t.schema.get_field_index('week_id'),t.schema.field('week_id'),pa.array(weeks[offset:offset+len(batch)],type=pa.int64()))
            assert actual.equals(expected),f'Value difference at label {label} row {offset}'
            offset+=len(batch)
        assert offset==len(s)
        outputs[str(label)]={'rows':offset,'sha256':digest(dest),'bytes':dest.stat().st_size}
        print('Label fully verified',label,offset,flush=True)
    idx=selected.rename(columns={'source_row':'source_dataset_row'}).reset_index(drop=True)
    idx['temporal_dataset_row']=np.arange(len(idx))
    pq.write_table(pa.Table.from_pandas(idx,preserve_index=False),out/'temporal_index.parquet',compression='zstd')
    provenance=idx[['sha256','temporal_dataset_row','label_output_row','week_id']].merge(pr.rename(columns={'dataset_row':'source_dataset_row'}),on='sha256',validate='one_to_one')
    assert len(provenance)==len(idx) and provenance.sha256.tolist()==idx.sha256.tolist()
    pq.write_table(pa.Table.from_pandas(provenance,preserve_index=False),out/'sample_provenance.parquet',compression='zstd')
    for name,frame in [('temporal_index.parquet',idx),('sample_provenance.parquet',provenance)]:
        assert pq.read_table(out/name).equals(pa.Table.from_pandas(frame,preserve_index=False))
    weeks=pd.Index(range(int(selected.week_id.max())+1),name='week_id')
    weekly=selected.groupby(['week_id','label']).size().unstack(fill_value=0).reindex(weeks,fill_value=0).rename(columns={0:'label_0',1:'label_1'}).reset_index()
    weekly['week_start_utc']=origin+pd.to_timedelta(weekly.week_id*7,unit='D')
    weekly['week_end_exclusive_utc']=weekly.week_start_utc+pd.Timedelta(days=7)
    weekly['total']=weekly.label_0+weekly.label_1
    weekly['label_1_percent']=weekly.label_1/weekly.total.replace(0,np.nan)*100
    weekly['label_0_percent']=weekly.label_0/weekly.total.replace(0,np.nan)*100
    weekly['cumulative_label_0']=weekly.label_0.cumsum();weekly['cumulative_label_1']=weekly.label_1.cumsum()
    csv(out,'weekly_counts.csv',weekly)
    full=pd.MultiIndex.from_product([weeks,['Win32','Win64','Dot_Net','OTHER_PE'],[0,1]],names=['week_id','file_type','label'])
    wg=selected.groupby(['week_id','file_type','label']).size().reindex(full,fill_value=0).reset_index(name='samples')
    wg=wg.merge(weekly[['week_id','week_start_utc','week_end_exclusive_utc']],on='week_id',validate='many_to_one')
    csv(out,'weekly_file_type_counts.csv',wg)
    selected['year']=selected.first_submission_utc.dt.year
    yearly=selected.groupby(['year','label']).size().unstack(fill_value=0).rename(columns={0:'label_0',1:'label_1'}).reset_index()
    yearly['total']=yearly.label_0+yearly.label_1;yearly['label_1_percent']=yearly.label_1/yearly.total*100
    csv(out,'yearly_counts.csv',yearly)
    assert weekly.label_0.sum()==196190 and weekly.label_1.sum()==553322
    assert wg.samples.sum()==len(selected) and yearly.total.sum()==len(selected)
    peak=weekly.loc[weekly.total.idxmax()]
    report={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),
            'source_rows':len(df),'included_rows':len(selected),'excluded_rows':len(excluded),
            'exclusion_reasons':excluded.exclusion_reason.value_counts().to_dict(),'outputs':outputs,
            'origin_utc':str(origin),'last_week_start_utc':str(weekly.week_start_utc.iloc[-1]),
            'week_count':len(weekly),'empty_weeks':int((weekly.total==0).sum()),
            'weeks_with_both_labels':int(((weekly.label_0>0)&(weekly.label_1>0)).sum()),
            'weeks_with_only_label_0':int(((weekly.label_0>0)&(weekly.label_1==0)).sum()),
            'weeks_with_only_label_1':int(((weekly.label_1>0)&(weekly.label_0==0)).sum()),
            'first_submission_min_utc':str(selected.first_submission_utc.min()),
            'first_submission_max_utc':str(selected.first_submission_utc.max()),
            'peak_week':{'week_id':int(peak.week_id),'start':str(peak.week_start_utc),'total':int(peak.total),'label_0':int(peak.label_0),'label_1':int(peak.label_1)},
            'all_rows_read_back_equal_to_source_except_week_id':True,
            'no_duplicate_sha256':True,'labels_disjoint':True,'all_week_boundaries_verified':True,
            'source_sha256':v['dataset_sha256'],'source_provenance_sha256':v['provenance_sha256'],
            'script_sha256':digest(Path(__file__)),'python':sys.version,'pyarrow':pa.__version__,
            'pandas':pd.__version__,'numpy':np.__version__,'elapsed_seconds':round(time.monotonic()-start,2),
            'balancing':False,'train_test_split':False,'training':False,'new_vt_requests':0}
    dump(out/'VALIDATION_REPORT.json',report);dump(out/'BUILD_STATE.json',{'status':'COMPLETE'})
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=BASE/'temporal_by_label_v1')
    parser.add_argument('--resume-finalization',action='store_true')
    args=parser.parse_args()
    main(args.output.resolve(),args.resume_finalization)
