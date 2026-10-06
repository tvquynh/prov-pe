"""Describe calendar-month counts without changing the temporal dataset."""
from pathlib import Path
import hashlib,json
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B=Path(__file__).resolve().parents[1]
S=B/'temporal_by_label_v1'
O=B/'monthly_distribution_20260927'
O.mkdir(exist_ok=False)
source=S/'temporal_index.parquet'
h=hashlib.sha256(source.read_bytes()).hexdigest()
checks=(S/'SHA256SUMS.txt').read_text().splitlines()
assert f'{h}  temporal_index.parquet' in checks
d=pq.read_table(source,columns=['sha256','label','file_type','first_submission_date']).to_pandas()
assert len(d)==749512 and d.sha256.is_unique
dt=pd.to_datetime(d.first_submission_date,unit='s',utc=True)
d['year_month']=dt.dt.strftime('%Y-%m')
months=pd.period_range(d.year_month.min(),d.year_month.max(),freq='M').astype(str)
w=d.groupby(['year_month','label']).size().unstack(fill_value=0).reindex(months,fill_value=0)
w.columns=['label_0','label_1'];w.index.name='year_month'
w['total']=w.label_0+w.label_1
w['label_1_percent']=100*w.label_1/w.total.replace(0,float('nan'))
w['month_id']=range(len(w))
w.to_csv(O/'monthly_counts.csv',encoding='utf-8-sig')
idx=pd.MultiIndex.from_product([months,sorted(d.file_type.unique()),[0,1]],names=['year_month','file_type','label'])
g=d.groupby(['year_month','file_type','label']).size().reindex(idx,fill_value=0).rename('samples')
g.to_csv(O/'monthly_file_type_counts.csv',encoding='utf-8-sig')
assert w.total.sum()==len(d)==g.sum()
assert w.label_0.sum()==196190 and w.label_1.sum()==553322
thin=w[(w.label_0<100)|(w.label_1<100)]
thin.to_csv(O/'months_with_either_label_below_100.csv',encoding='utf-8-sig')
summary={'status':'PASS','rows':len(d),'source_sha256':h,'months':len(w),'first_month':months[0],'last_month':months[-1],
 'empty_months':int((w.total==0).sum()),'both_labels':int(((w.label_0>0)&(w.label_1>0)).sum()),
 'only_label_0':int(((w.label_0>0)&(w.label_1==0)).sum()),'only_label_1':int(((w.label_1>0)&(w.label_0==0)).sum()),
 'either_label_below_100':len(thin),'total_below_100':int((w.total<100).sum()),
 'monthly_total_median':float(w.total.median()),'peak_month':w.total.idxmax(),'peak_counts':w.loc[w.total.idxmax()].to_dict(),
 'file_types':sorted(d.file_type.unique()),'membership_changed':False,'train_test_split':False}
summary['periods']=[]
for name,start,end in [('2006-2019','2006-01','2019-12'),('2020-2023','2020-01','2023-12'),('2024','2024-01','2024-12'),('2025','2025-01','2025-12'),('2026-01 to 2026-09','2026-01','2026-09')]:
 z=w.loc[start:end]
 summary['periods'].append({'period':name,'months':len(z),'label_0':int(z.label_0.sum()),'label_1':int(z.label_1.sum()),'total':int(z.total.sum()),'min_label_0':int(z.label_0.min()),'min_label_1':int(z.label_1.min())})
(O/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
x=pd.to_datetime(w.index)
fig,axs=plt.subplots(3,1,figsize=(12,9),layout='constrained')
for k,c in [('label_0','#4379a9'),('label_1','#b96b38')]:
 axs[0].plot(x,w[k],label=k,color=c,lw=1)
 axs[1].plot(x,w[k],label=k,color=c,lw=1)
axs[0].set_ylabel('Samples per month');axs[0].legend(frameon=False)
axs[1].set_yscale('symlog',linthresh=1);axs[1].set_ylabel('Samples (symlog scale)')
axs[2].plot(x,w.label_1_percent,color='#b96b38');axs[2].set_ylim(0,100);axs[2].set_ylabel('Label 1 (%)')
axs[2].set_xlabel('First submission month (UTC)')
for ax in axs:ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.savefig(O/'monthly_distribution.png',dpi=160);fig.savefig(O/'monthly_distribution.svg');plt.close(fig)
assert hashlib.sha256(source.read_bytes()).hexdigest()==h
print(json.dumps(summary,ensure_ascii=False,indent=2))
