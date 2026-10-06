"""Summarize annual class counts and within-year monthly coverage."""
from pathlib import Path
import hashlib,json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]
O=B/'yearly_distribution_20260927'
O.mkdir(exist_ok=False)
src=B/'monthly_distribution_20260927/monthly_counts.csv'
m=pd.read_csv(src);m['year']=m.year_month.str[:4].astype(int)
y=m.groupby('year')[['label_0','label_1','total']].sum()
y['label_1_percent']=100*y.label_1/y.total
y['observed_calendar_months']=m.groupby('year').size()
y['months_both_labels']=m.assign(both=(m.label_0>0)&(m.label_1>0)).groupby('year').both.sum()
y['min_month_label_0']=m.groupby('year').label_0.min()
y['min_month_label_1']=m.groupby('year').label_1.min()
y['peak_month_share_percent']=100*m.groupby('year').total.max()/y.total
y['partial_boundary_year']=y.index.isin([2006,2026])
old=pd.read_csv(B/'temporal_by_label_v1/yearly_counts.csv').set_index('year')
pd.testing.assert_frame_equal(y[['label_0','label_1','total']],old[['label_0','label_1','total']])
assert int(y.total.sum())==749512
y.to_csv(O/'yearly_distribution.csv',encoding='utf-8-sig')
g=pd.read_csv(B/'monthly_distribution_20260927/monthly_file_type_counts.csv')
g['year']=g.year_month.str[:4].astype(int)
gt=g.groupby(['year','file_type','label']).samples.sum()
assert gt.sum()==749512
gt.to_csv(O/'yearly_file_type_counts.csv',encoding='utf-8-sig')
fig,ax=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
ax[0].bar(y.index-.2,y.label_0,width=.4,label='Label 0',color='#4379a9')
ax[0].bar(y.index+.2,y.label_1,width=.4,label='Label 1',color='#b96b38')
ax[0].set_ylabel('Samples');ax[0].legend(frameon=False)
ax[1].plot(y.index,y.label_1_percent,'o-',color='#b96b38');ax[1].set_ylim(0,100)
ax[1].set_ylabel('Label 1 (%)');ax[1].set_xlabel('First submission year (UTC); 2006 and 2026 are partial years')
ax[1].set_xticks(y.index);ax[1].tick_params(axis='x',rotation=45)
for a in ax:a.grid(axis='y',alpha=.2);a.spines[['top','right']].set_visible(False)
fig.savefig(O/'yearly_distribution.png',dpi=160);fig.savefig(O/'yearly_distribution.svg');plt.close(fig)
summary={'status':'PASS','rows':int(y.total.sum()),'years':len(y),'years_both_labels':int(((y.label_0>0)&(y.label_1>0)).sum()),'years_either_label_below_100':int(((y.label_0<100)|(y.label_1<100)).sum()),'years_either_label_below_1000':int(((y.label_0<1000)|(y.label_1<1000)).sum()),'matches_existing_annual_table':True,'monthly_input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'membership_changed':False,'train_test_split':False}
(O/'validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary));print(y.to_string())
