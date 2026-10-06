"""Plot the completed temporal descriptive tables without changing membership."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]/'temporal_by_label_v1'
w=pd.read_csv(P/'weekly_counts.csv');y=pd.read_csv(P/'yearly_counts.csv')
x=pd.to_datetime(w.week_start_utc,utc=True)
fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
for label,color in [(0,'#4379a9'),(1,'#b96b38')]:
    axes[0].plot(x,w[f'label_{label}'],label=f'Label {label}',color=color,linewidth=.9)
    axes[1].plot(x,w[f'cumulative_label_{label}'],label=f'Label {label}',color=color,linewidth=1.5)
axes[0].set_ylabel('Samples per UTC week');axes[0].legend(frameon=False)
axes[1].set_ylabel('Cumulative samples');axes[1].set_xlabel('First submission to VirusTotal (UTC)')
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
fig.savefig(P/'figures/weekly_counts_and_cumulative.png',dpi=180)
fig.savefig(P/'figures/weekly_counts_and_cumulative.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
ax.bar(y.year-.2,y.label_0,width=.4,label='Label 0',color='#4379a9')
ax.bar(y.year+.2,y.label_1,width=.4,label='Label 1',color='#b96b38')
ax.set_xticks(y.year);ax.tick_params(axis='x',rotation=60)
ax.set_xlabel('Year of first submission to VirusTotal (UTC)');ax.set_ylabel('Samples')
ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
fig.savefig(P/'figures/yearly_counts.png',dpi=180);fig.savefig(P/'figures/yearly_counts.svg');plt.close(fig)
print('Figures created')
