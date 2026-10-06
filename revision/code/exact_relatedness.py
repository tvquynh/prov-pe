"""Exact existence search at TLSH distance <=30 using nonnegative header bounds.

No triangle inequality, approximate neighbor index, learned representation or labels
are used. A witness is the first qualifying source, not a nearest-neighbor claim.
"""
import os
os.environ['NUMBA_NUM_THREADS']='16'
import time
import numpy as np,pandas as pd,tlsh,numba
from numba import njit,prange
from revision_common import *

def decode(hashes):
    a=np.frombuffer(b''.join(bytes.fromhex(h[2:] if h.startswith('T1') else h) for h in hashes),dtype=np.uint8).reshape(-1,35).copy()
    # The displayed checksum and length bytes have swapped nibbles.
    b=np.empty((len(a),36),np.uint8)
    b[:,0]=(a[:,0]>>4)|((a[:,0]&15)<<4)
    b[:,1]=(a[:,1]>>4)|((a[:,1]&15)<<4)
    b[:,2]=a[:,2]>>4;b[:,3]=a[:,2]&15;b[:,4:]=a[:,3:]
    return b

def lookup():
    a=np.arange(256,dtype=np.int32)[:,None];b=np.arange(256,dtype=np.int32)[None,:]
    lut=np.zeros((256,256),np.uint8)
    for shift in (0,2,4,6):
        d=np.abs(((a>>shift)&3)-((b>>shift)&3));lut+=np.where(d==3,6,d).astype(np.uint8)
    return lut

@njit(cache=True)
def distance(a,b,lut):
    dl=abs(np.int32(a[1])-np.int32(b[1]));dl=min(dl,256-dl)
    total=dl if dl<=1 else 12*dl
    for k in (2,3):
        dd=abs(np.int32(a[k])-np.int32(b[k]));dd=min(dd,16-dd)
        total+=dd if dd<=1 else 12*(dd-1)
    total+=int(a[0]!=b[0])
    for k in range(4,36):total+=np.int32(lut[a[k],b[k]])
    return total

@njit(parallel=True,cache=True)
def search(queries,source,offset,lut):
    witness=np.full(len(queries),-1,np.int64);ds=np.full(len(queries),-1,np.int16);comparisons=np.zeros(len(queries),np.int64)
    for i in prange(len(queries)):
        q=queries[i];found=False
        for dl in range(-2,3):
            pl=abs(dl) if abs(dl)<=1 else 12*abs(dl)
            for d1 in range(-3,4):
                p1=abs(d1) if abs(d1)<=1 else 12*(abs(d1)-1)
                for d2 in range(-3,4):
                    p2=abs(d2) if abs(d2)<=1 else 12*(abs(d2)-1)
                    bound=pl+p1+p2
                    if bound>30:continue
                    key=((np.int32(q[1])+dl)%256)*256+((np.int32(q[2])+d1)%16)*16+(np.int32(q[3])+d2)%16
                    for j in range(offset[key],offset[key+1]):
                        s=source[j];total=bound+int(q[0]!=s[0]);comparisons[i]+=1
                        for k in range(4,36):
                            total+=np.int32(lut[q[k],s[k]])
                            if total>30:break
                        if total<=30:
                            witness[i]=j;ds[i]=total;found=True;break
                    if found:break
                if found:break
            if found:break
    return witness,ds,comparisons

def index(a):
    key=a[:,1].astype(np.int32)*256+a[:,2].astype(np.int32)*16+a[:,3]
    order=np.argsort(key,kind='stable');off=np.r_[0,np.cumsum(np.bincount(key,minlength=65536))]
    return a[order],off,order

def qualify(lut):
    rng=np.random.default_rng(17091) # software test randomness, not training seeds
    hs=['T1'+bytes(rng.integers(0,256,35,dtype=np.uint8)).hex().upper() for _ in range(400)]
    aa=decode(hs);checks=0
    for i in range(20000):
        x,y=rng.integers(0,len(hs),2)
        assert distance(aa[x],aa[y],lut)==tlsh.diff(hs[x],hs[y]);checks+=1
    # Include close mutants so both sides of the decision boundary are exercised.
    mutated=[]
    for h in hs:
        raw=bytearray.fromhex(h[2:])
        for _ in range(int(rng.integers(1,5))):raw[int(rng.integers(0,35))]^=int(rng.integers(1,256))
        mutated.append('T1'+raw.hex().upper())
    all_h=hs+mutated;all_a=decode(all_h);ss,off,order=index(aa)
    w,dd,cc=search(all_a,ss,off,lut)
    for i,h in enumerate(all_h):
        expected=any(tlsh.diff(h,s)<=30 for s in hs)
        assert (w[i]>=0)==expected
        if w[i]>=0:assert int(dd[i])==tlsh.diff(h,hs[order[w[i]]])
    return {'random_distance_checks':checks,'exhaustive_library_comparison_queries':len(all_h),'reference_hashes_per_query':len(hs),'status':'PASS'}

def main():
    out=ROOT/'data/exact_source_relatedness.parquet';assert not out.exists()
    save(ROOT/'config/relatedness_specification.json',{'recorded_utc':datetime.now(timezone.utc).isoformat(),'threshold':30,'metric':'py-tlsh 5.0.0 diff including length','search':'Exact existence with header lower-bound pruning and nonnegative body early exit','source_scope':'All assessable canonical TLSH digests in official PE train, test and Challenge manifests','target_scope':'All primary rows, independent of collection source and label','no_approximation':True,'witness':'first source within threshold, not minimum distance','code_sha256':sha(Path(__file__)),'software_test_seed':17091})
    lut=lookup();qa=qualify(lut);save(ROOT/'logs/relatedness_qualification.json',qa);log('relatedness_qualification_passed',**qa)
    sources=[];identities={}
    for split in ['train','test','challenge']:
        p=ORIGINAL/f'data/source_{split}_manifest.parquet';identities[split]=sha(p)
        z=pd.read_parquet(p,columns=['sha256','tlsh_canonical']);z['source_split']=split;sources.append(z)
    full=pd.concat(sources,ignore_index=True)
    source=full.dropna(subset=['tlsh_canonical']).sort_values(['tlsh_canonical','sha256']).drop_duplicates('tlsh_canonical').reset_index(drop=True)
    a,off,order=index(decode(source.tlsh_canonical.tolist()));source=source.iloc[order].reset_index(drop=True)
    source.to_parquet(ROOT/'data/relatedness_reference_digest_index.parquet',index=False)
    target=pd.read_parquet(ORIGINAL/'data/evaluation_manifest.parquet',columns=['sha256','tlsh_canonical'])
    unique=np.sort(target.tlsh_canonical.unique());q=decode(unique)
    # Equality lookup avoids unnecessary scans; other queries use complete header bounds.
    eq=pd.Index(source.tlsh_canonical).get_indexer(unique)
    witness=eq.astype(np.int64);dist=np.where(eq>=0,0,-1).astype(np.int16);counts=np.zeros(len(q),np.int64)
    todo=np.flatnonzero(eq<0);started=time.monotonic()
    for first in range(0,len(todo),5000):
        ids=todo[first:first+5000];w,d,c=search(q[ids],a,off,lut);witness[ids]=w;dist[ids]=d;counts[ids]=c
        log('relatedness_progress',done=min(first+5000,len(todo)),total=len(todo),seconds=time.monotonic()-started)
    valid=witness>=0
    for i in np.flatnonzero(valid):assert tlsh.diff(unique[i],source.tlsh_canonical.iloc[witness[i]])==int(dist[i])
    matches=pd.DataFrame({'tlsh_canonical':unique,'has_source_neighbor_le30':valid,'witness_distance':dist,'header_eligible_comparisons':counts,'reference_sha256':[source.sha256.iloc[j] if j>=0 else None for j in witness],'reference_tlsh':[source.tlsh_canonical.iloc[j] if j>=0 else None for j in witness]})
    result=target.merge(matches,on='tlsh_canonical',how='left',validate='many_to_one',sort=False)
    assert np.array_equal(result.sha256,target.sha256) and result.has_source_neighbor_le30.notna().all()
    result.to_parquet(out,index=False)
    save(ROOT/'logs/exact_relatedness_complete.json',{'status':'COMPLETE','target_rows':len(target),'target_unique_digests':len(unique),'source_rows':len(full),'source_unassessable_rows':int(full.tlsh_canonical.isna().sum()),'source_unique_assessable_digests':len(source),'target_with_neighbor':int(result.has_source_neighbor_le30.sum()),'compared_pairs':int(counts.sum()),'elapsed_seconds':time.monotonic()-started,'all_witnesses_library_verified':True,'source_hashes':identities,'ledger_sha256':sha(out),'library_version':'5.0.0','numba_version':numba.__version__})
    log('exact_relatedness_complete',affected=int(result.has_source_neighbor_le30.sum()))
if __name__=='__main__':main()
