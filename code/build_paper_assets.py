"""Generate all numerical TeX cells and figures from retained audit/results files."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from common import *

NAMES={'source_test':'Source test','external_primary':'External primary','external_temporal':'Temporal condition',
       'external_tls_equal':'Identical-TLSH control','external_temporal_tls_equal':'Both conditions','external_known_near_control':'Verified-neighbor control'}
SHORT={'source_test':'SRC','external_primary':'PRI','external_temporal':'W30','external_tls_equal':'TLS','external_temporal_tls_equal':'BOTH','external_known_near_control':'NEAR'}


def tex(path,text):
    path.write_text(text+'\n',encoding='utf8')


def cell(row,metric):
    if pd.isna(row[metric+'_mean']):
        return r'\textit{n/a}'
    precision=3 if metric in ['FPR','ROC_AUC','AP'] else 2
    return f'${100*row[metric+"_mean"]:.{precision}f}\\pm{100*row[metric+"_sd"]:.{precision}f}$'


def main():
    paper=ROOT/'paper'
    generated=paper/'generated'
    figdir=paper/'figures'
    audit=read(ROOT/'qualification/input_audit.json')
    summary=pd.read_csv(ROOT/'results/summary_results.csv')
    individual=pd.read_csv(ROOT/'results/per_seed_results.csv')
    subgroup=pd.read_csv(ROOT/'results/subgroup_summary.csv')
    pairs=pd.read_csv(ROOT/'results/paired_differences.csv')
    target_path=ROOT/'data/evaluation_manifest.parquet'
    if not target_path.exists():target_path=ROOT/'review_data/target.parquet'
    target=pd.read_parquet(target_path,columns=['label'])
    source_summary=read(ROOT/'descriptive_inputs/summary.json')
    extra=read(ROOT/'descriptive_inputs/additional_summary.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
    def row(cohort,alpha=.01,variant='prefix2480'):
        r=summary.loc[(summary.cohort==cohort)&(summary.source_fpr_target==alpha)&(summary.variant==variant)]
        assert len(r)==1
        return r.iloc[0]
    def savefig(fig,name):
        fig.savefig(figdir/(name+'.pdf'))
        fig.savefig(figdir/(name+'.png'),dpi=220)
        plt.close(fig)
    macros={
        'ExtTprOne':100*row('external_primary').TPR_mean,'ExtFprOne':100*row('external_primary').FPR_mean,
        'SrcTprOne':100*row('source_test').TPR_mean,'SrcFprOne':100*row('source_test').FPR_mean,
        'ExtTprTenth':100*row('external_primary',.001).TPR_mean,'ExtFprTenth':100*row('external_primary',.001).FPR_mean,
        'ExtAuc':100*row('external_primary').ROC_AUC_mean,'ExtAp':100*row('external_primary').AP_mean,
        'TempFprOne':100*row('external_temporal').FPR_mean,'TlsFprOne':100*row('external_tls_equal').FPR_mean,
        'TlsTprOne':100*row('external_tls_equal').TPR_mean,'PrimaryPositiveFraction':100*target.label.eq(1).mean(),
        'WarnDeltaTprOne':100*(row('external_primary',variant='full2568').TPR_mean-row('external_primary').TPR_mean),
        'WarnDeltaFprOne':100*(row('external_primary',variant='full2568').FPR_mean-row('external_primary').FPR_mean)}
    macros['NearFprOne']=100*row('external_known_near_control').FPR_mean
    macros['NearTprOne']=100*row('external_known_near_control').TPR_mean
    macros['SrcTprTenth']=100*row('source_test',.001).TPR_mean
    macros['SrcFprTenth']=100*row('source_test',.001).FPR_mean
    for value,prefix in [('0','NonDll'),('1','Dll')]:
        sub=subgroup[(subgroup.variant=='prefix2480')&(subgroup.source_fpr_target==.01)&(subgroup.grouping=='is_dll')&(subgroup['group'].astype(str)==value)].iloc[0]
        macros[prefix+'FprOne']=100*sub.FPR_mean
        macros[prefix+'TprOne']=100*sub.TPR_mean
    sub=subgroup[(subgroup.variant=='prefix2480')&(subgroup.source_fpr_target==.01)&(subgroup.grouping=='file_type_by_is_dll')&(subgroup['group']=='Dot_Net;DLL=0')].iloc[0]
    macros['DotNetNonDllFprOne']=100*sub.FPR_mean
    tex(generated/'results_macros.tex','\n'.join('\\newcommand{\\'+k+'}{'+format(v,'.3f' if 'Fpr' in k else '.2f')+'}' for k,v in macros.items()))
    lines=[r'\begin{table}[pos=t]',r'\centering\small',r'\begin{tabular}{lrr}',r'\toprule',r'Cohort & Label 0 & Label 1 \\',r'\midrule']
    for cohort in NAMES:
        r=row(cohort)
        lines.append(f'{NAMES[cohort]} & {r.n_negative:,} & {r.n_positive:,} '+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Class counts in the source-test reference and fixed external cohorts. Labels in the external cohorts follow the stated operational rule.}',r'\label{tab:cohorts}',r'\end{table}']
    tex(generated/'cohort_table.tex','\n'.join(lines))
    lines=[r'\begin{table*}[pos=t]',r'\centering\scriptsize',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
           r'& & & \multicolumn{2}{c}{1\% source target} & \multicolumn{2}{c}{0.1\% source target} \\',
           r'Cohort & ROC-AUC & AP & TPR & FPR & TPR & FPR \\',r'\midrule']
    for cohort in NAMES:
        a,b=row(cohort),row(cohort,.001)
        lines.append(NAMES[cohort]+' & '+' & '.join([cell(a,'ROC_AUC'),cell(a,'AP'),cell(a,'TPR'),cell(a,'FPR'),cell(b,'TPR'),cell(b,'FPR')])+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Primary warning-free reference: five-seed mean $\pm$ sample SD. All metrics are percentages. Targets refer to source-validation FPR; the FPR columns report rates realized on the named evaluation cohort.}',r'\label{tab:performance}',r'\end{table*}']
    tex(generated/'performance_table.tex','\n'.join(lines))
    lines=[r'\begin{table*}[pos=t]',r'\centering\small',r'\begin{tabular}{llrrrr}',r'\toprule',r'File group & DLL flag & Label 0 & Label 1 & TPR (\%) & FPR (\%) \\',r'\midrule']
    selected=subgroup[(subgroup.variant=='prefix2480')&(subgroup.source_fpr_target==.01)&(subgroup.grouping=='file_type_by_is_dll')]
    for _,r in selected.iterrows():
        kind,dll=r['group'].split(';DLL=')
        lines.append(kind.replace('Dot_Net','.NET')+' & '+dll+' & '+f'{r.n_negative:,} & {r.n_positive:,} & '+cell(r,'TPR')+' & '+cell(r,'FPR')+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Primary external cohort by file group and recorded DLL characteristic at the 1\% source-validation target. Rates are mean $\pm$ sample SD across five seeds; class counts are fixed. The DLL flag denotes the PE header characteristic, not observed execution.}',r'\label{tab:subgroups}',r'\end{table*}']
    tex(generated/'subgroup_table.tex','\n'.join(lines))
    lines=[r'\begin{table}[pos=t]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{llrr}',r'\toprule',r'Cohort & Target & $\Delta$TPR & $\Delta$FPR \\',r'\midrule']
    for cohort in ['source_test','external_primary']:
        for alpha in [.01,.001]:
            p=pairs[(pairs.comparison=='full2568 minus prefix2480')&(pairs.variant==cohort)&(pairs.source_fpr_target==alpha)]
            assert len(p)==5
            diffs=[f'${p[m].mean():+.{3 if m.startswith("FPR") else 2}f}\\pm{p[m].std(ddof=1):.{3 if m.startswith("FPR") else 2}f}$' for m in ['TPR_difference_pp','FPR_difference_pp']]
            lines.append(NAMES[cohort]+' & '+f'{100*alpha:g}'+r'\% & '+' & '.join(diffs)+r' \\')
    lines += [r'\bottomrule\end{tabular}',r'\caption{Full-feature minus warning-free paired differences, in percentage points. Each model selects its own threshold on the corresponding source-validation subset. Values are mean $\pm$ sample SD of five paired differences.}',r'\label{tab:representation}',r'\end{table}']
    tex(generated/'representation_table.tex','\n'.join(lines))
    lines=[r'\small',r'\begin{longtable}{lllrrrrr}',r'\caption{All seed results. P denotes the primary projection and F the full-feature sensitivity. SRC, PRI, W30, TLS, BOTH and NEAR denote source test, primary, temporal, identical-TLSH, intersection and verified-neighbor views. Metrics are percentages.}\\',r'\toprule',r'View & Seed & Cohort & Target & AUC & AP & TPR & FPR \\',r'\midrule\endfirsthead',r'\toprule',r'View & Seed & Cohort & Target & AUC & AP & TPR & FPR \\',r'\midrule\endhead']
    for _,r in individual.sort_values(['variant','seed','cohort','source_fpr_target']).iterrows():
        lines.append(('P' if r.variant=='prefix2480' else 'F')+' & '+str(int(r.seed))+' & '+SHORT[r.cohort]+' & '+f'{100*r.source_fpr_target:g} & {100*r.ROC_AUC:.3f} & {100*r.AP:.3f} & {100*r.TPR:.3f} & {100*r.FPR:.3f}'+r' \\')
    lines += [r'\bottomrule\end{longtable}',r'\normalsize']
    tex(generated/'all_seed_tables.tex','\n'.join(lines))
    thresholds=pd.read_csv(ROOT/'results/thresholds.csv')
    lines=[r'\begin{longtable}{llrrr}',r'\toprule',r'View & Seed & Source target (\%) & Threshold & Actual validation FPR (\%) \\',r'\midrule\endhead']
    for _,r in thresholds.iterrows():
        lines.append(('P' if r.variant=='prefix2480' else 'F')+f' & {int(r.seed)} & {100*r.source_fpr_target:g} & {r.threshold:.12f} & {100*r.validation_fpr:.5f}'+r' \\')
    lines += [r'\bottomrule\end{longtable}']
    tex(generated/'threshold_table.tex','\n'.join(lines))
    # Counts are derived, never independently typed into figure labels.
    total=source_summary['quality']['rows']
    count_labels={r['label']:r['samples'] for r in source_summary['labels']}
    assert sum(count_labels.values())==total==779619
    metadata=read(ROOT/'qualification/metadata_audit_verified.json')
    valid=metadata['denominator_terminal_ok']
    ages=np.array([valid-extra['reports_over_30_days_old'],extra['reports_over_30_days_old']-extra['reports_over_365_days_old'],extra['reports_over_365_days_old']])
    assert ages.sum()==valid
    fig,ax=plt.subplots(figsize=(3.45,2.6))
    ax.bar(np.arange(3),ages/valid*100,color=['#bac5ce','#7a92a5','#365b78'],width=.6)
    for x,n in enumerate(ages):ax.text(x,100*n/valid+1.5,f'{n:,}\n{100*n/valid:.2f}%',ha='center',fontsize=8)
    ax.set_xticks(range(3),['[0, 30]','(30, 365]','>365'])
    ax.set(xlabel='Age of stored analysis (days)',ylabel='Usable reports (%)',ylim=(0,82))
    fig.tight_layout();savefig(fig,'report_age')
    fig,ax=plt.subplots(figsize=(7.1,3.55));ax.set(xlim=(0,10),ylim=(0,5));ax.axis('off')
    steps=[('Inventory',total)]+[(r['condition'],r['after']) for r in audit['sequential_exclusions']]
    names=['Retrieved\ninventory','Binary\nlabels','Three\nPE groups','First submission\navailable','SHA-disjoint\nprimary cohort']
    for i,((_,n),name) in enumerate(zip(steps,names)):
        x=i*2.0
        ax.add_patch(Rectangle((x+.05,2.85),1.82,1.15,facecolor='white',edgecolor='#365b78'))
        ax.text(x+.96,3.42,name+f'\n{n:,}',ha='center',va='center',fontsize=7.8)
        if i<4:ax.add_patch(FancyArrowPatch((x+1.87,3.42),(x+2.04,3.42),arrowstyle='-|>',mutation_scale=8))
        if i>0:ax.text(x+.96,2.52,f'Excluded: {audit["sequential_exclusions"][i-1]["excluded"]:,}',ha='center',fontsize=7.7)
    for x,(name,key) in zip([.4,3.6,6.8],[('Temporal condition','temporal_sensitivity'),('Identical-TLSH control','identical_tlsh_sensitivity'),('Both conditions','temporal_and_tlsh_sensitivity')]):
        r=audit['cohorts'][key]
        ax.add_patch(Rectangle((x,.3),2.8,1.05,facecolor='#f0f3f5',edgecolor='#7a92a5'))
        ax.text(x+1.4,.82,f'{name}\n{r["rows"]:,} records',ha='center',va='center',fontsize=8.5)
    ax.text(5,1.85,'Fixed subsets of the primary cohort; the original inventory is retained',ha='center',fontsize=8.5)
    savefig(fig,'cohort_flow')
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.0))
    cohorts=['source_test','external_primary']
    for ax,metric in zip(axes,['TPR','FPR']):
        for j,alpha in enumerate([.01,.001]):
            color=['#365b78','#9b673d'][j]
            for i,cohort in enumerate(cohorts):
                vals=individual[(individual.variant=='prefix2480')&(individual.cohort==cohort)&(individual.source_fpr_target==alpha)][metric].to_numpy()*100
                xpos=i+(-.13 if j==0 else .13)
                ax.scatter(xpos+np.linspace(-.03,.03,5),vals,s=13,color=color,alpha=.6)
                ax.errorbar(xpos,vals.mean(),yerr=vals.std(ddof=1),fmt='s',color=color,capsize=3,label=f'{100*alpha:g}% source target' if i==0 else None)
            if metric=='FPR':ax.axhline(100*alpha,color=color,linestyle=':',lw=.8)
        ax.set_xticks(range(2),['Source test','External primary'])
        ax.set_ylabel(metric+' (%)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.legend(fontsize=7,loc='best',frameon=False)
    fig.tight_layout();savefig(fig,'threshold_transfer')
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.0))
    cohorts=list(NAMES)[1:]
    for ax,metric in zip(axes,['TPR','FPR']):
        for j,alpha in enumerate([.01,.001]):
            values=np.array([row(c,alpha)[metric+'_mean']*100 for c in cohorts])
            errors=np.array([row(c,alpha)[metric+'_sd']*100 for c in cohorts])
            ax.errorbar(np.arange(len(cohorts)),values,yerr=errors,fmt='o-',capsize=3,color=['#365b78','#9b673d'][j],label=f'{100*alpha:g}% source target',markersize=4,lw=1)
        ax.set_xticks(range(len(cohorts)),['Primary','Temporal','TLSH','Both','Neighbor'],rotation=15)
        ax.set_ylabel(metric+' (%)');ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=7)
    fig.tight_layout();savefig(fig,'eligibility_sensitivity')
    identities={str(p.relative_to(ROOT)):sha(p) for p in list(generated.glob('*.tex'))+list(figdir.glob('*.pdf'))}
    save(ROOT/'results/paper_asset_receipt.json',{'status':'GENERATED_FROM_RETAINED_RESULTS','code_sha256':sha(Path(__file__)),
                                                'input_results_sha256':sha(ROOT/'results/summary_results.csv'),'numeric_macros':macros,'output_identities':identities})
    log('paper_assets_complete')


if __name__=='__main__':
    main()
