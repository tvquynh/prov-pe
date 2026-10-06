"""Generate revision manuscript numbers, tables and figures from saved results."""
from pathlib import Path
import re,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from revision_common import ROOT,SEEDS,read,save,sha
PAPER=ROOT.parent/'paper';G=PAPER/'generated';FIG=PAPER/'figures'
def write(name,text):
    if '\u2014' in text:raise ValueError('Forbidden EM DASH')
    (G/name).write_text(text.strip()+'\n',encoding='utf8')
def fmt(row,metric,dp=2):return f"{100*row[metric+'_mean']:.{dp}f} $\\pm$ {100*row[metric+'_sd']:.{dp}f}"
def row(df,**kwargs):
    mask=np.ones(len(df),bool)
    for k,v in kwargs.items():mask&=df[k].eq(v)
    r=df.loc[mask];assert len(r)==1;return r.iloc[0]
def build():
    source=pd.read_csv(ROOT/'results/source_diagnostic_summary.csv')
    linear=pd.read_csv(ROOT/'results/linear_summary.csv')
    age=pd.read_csv(ROOT/'results/label_by_age_summary.csv')
    cut=pd.read_csv(ROOT/'results/label_sensitivity_summary.csv')
    common=pd.read_csv(ROOT/'results/common_support_summary.csv')
    near=pd.read_csv(ROOT/'results/exact_relatedness_summary.csv')
    old=pd.read_csv(ROOT.parent/'results/summary_results.csv')
    sd=row(source,group='all');l1=row(linear,group='primary',source_fpr_target=.01);l01=row(linear,group='primary',source_fpr_target=.001)
    nr=row(near,source_fpr_target=.01);nr01=row(near,source_fpr_target=.001)
    macros={'SourceAuc':100*sd.ROC_AUC_mean,'LinearTprOne':100*l1.TPR_mean,'LinearFprOne':100*l1.FPR_mean,'LinearTprTenth':100*l01.TPR_mean,'LinearFprTenth':100*l01.FPR_mean,'LinearAuc':100*l1.ROC_AUC_mean,'SymTprOne':100*nr.TPR_mean,'SymFprOne':100*nr.FPR_mean,'SymTprTenth':100*nr01.TPR_mean,'SymFprTenth':100*nr01.FPR_mean}
    write('revision_macros.tex','\n'.join('\\newcommand{\\'+k+'}{'+f'{v:.3f}'+'}' for k,v in macros.items()))
    rows=[]
    for key,label in [('all','All test records'),('VT_label_0','Recorded VT label 0'),('VT_label_1','Recorded VT label 1')]:
        r=row(source,group=key)
        rows.append(label+' & '+ ' & '.join(fmt(r,m,2) for m in ['ROC_AUC','AP','TPR','TNR','balanced_accuracy'])+r' \\')
    write('source_diagnostic_main.tex',r'''\begin{table*}[pos=t]
\centering\small
\begin{tabular}{lrrrrr}
\toprule
Held-out source-prediction view & ROC-AUC & AP & Sensitivity & Specificity & Balanced accuracy \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}
\caption{Collection-source prediction from the warning-free representation, expressed as percentage mean $\pm$ sample SD over five group-separated splits. The positive source is MalwareBazaar. These are source-identification metrics, not malware-detection rates. Within-label metrics condition the test set only; the diagnostic models were fitted to pooled source records.}
\label{tab:source-diagnostic}
\end{table*}''')
    rows=[]
    for key,label in [('0-30','0--30'),('31-365','31--365'),('over365','$>365$')]:
        a=row(age,positive_minimum=5,report_age=key,source_fpr_target=.01);b=row(age,positive_minimum=5,report_age=key,source_fpr_target=.001)
        rows.append(label+f' & {int(a.n_negative_min):,} & {int(a.n_positive_min):,} & '+fmt(a,'TPR')+' & '+fmt(a,'FPR',3)+' & '+fmt(b,'FPR',3)+r' \\')
    write('report_age_main.tex',r'''\begin{table*}[pos=t]
\centering\small
\begin{tabular}{lrrr rr}
\toprule
Report age (days) & Label 0 & Label 1 & TPR at 1\% & FPR at 1\% & FPR at 0.1\% \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}
\caption{Primary LightGBM performance by stored report age. Rates are percentage mean $\pm$ sample SD over five seeds; column targets refer to source-validation FPR. Age is the time from analysis to report retrieval.}
\label{tab:report-age}
\end{table*}''')
    rows=[]
    for k in [5,10,15]:
        a=row(cut,positive_minimum=k,source_fpr_target=.01);b=row(cut,positive_minimum=k,source_fpr_target=.001)
        rows.append(f'$m\\geq{k}$ & {int(a.n_positive_min):,} & '+fmt(a,'TPR')+' & '+fmt(b,'TPR')+' & '+fmt(a,'ROC_AUC',3)+r' \\')
    write('label_sensitivity_main.tex',r'''\begin{table*}[pos=t]
\centering\small
\begin{tabular}{lrrrr}
\toprule
Positive rule & Positive records & TPR at 1\% & TPR at 0.1\% & ROC-AUC \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}
\caption{VT-count sensitivity using the unchanged LightGBM scores and thresholds. All views retain the same 189,114 label-0 records; their FPRs are therefore identical (0.358\% and 0.045\%). Entries are percentage mean $\pm$ sample SD across five seeds. The alternative cutoffs were added during revision.}
\label{tab:label-sensitivity}
\end{table*}''')
    rows=[]
    views=[('restricted_unweighted','pooled','Restricted, observed mixture'),('source_standardized','benign_reference_candidate','Standardized benign-reference'),('source_standardized','mb_malware_candidate','Standardized MalwareBazaar'),('source_standardized','pooled','Standardized equal-source mixture')]
    for view,source_name,label in views:
        a=row(common,view=view,source=source_name,source_fpr_target=.01);b=row(common,view=view,source=source_name,source_fpr_target=.001)
        rows.append(label+' & '+fmt(a,'FPR',3)+' & '+fmt(b,'FPR',3)+r' \\')
    write('support_main.tex',r'''\begin{table*}[pos=t]
\centering\small
\begin{tabular}{lrr}
\toprule
Label-0 evaluation view & FPR at 1\% & FPR at 0.1\% \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}
\caption{FPR on the 160,679 negative records in common structural support. Entries are percentage mean $\pm$ sample SD. Standardized views use identical cell weights; the pooled standardized view gives equal mass to each source. No label-1 cell satisfies the support rule.}
\label{tab:support}
\end{table*}''')
    rows=[]
    for group,name in [('source_test','Source test'),('primary','External primary'),('exact_TLSH_le30_excluded','Symmetric distance control')]:
        for alpha in [.01,.001]:
            r=row(linear,group=group,source_fpr_target=alpha)
            rows.append(name+f' & {100*alpha:g} & '+fmt(r,'TPR')+' & '+fmt(r,'FPR',3)+' & '+fmt(r,'ROC_AUC',3)+' & '+fmt(r,'AP',3)+r' \\')
    write('linear_main.tex',r'''\begin{table*}[pos=t]
\centering\small
\begin{tabular}{llrrrr}
\toprule
Logistic Regression evaluation & Source target (\%) & TPR & FPR & ROC-AUC & AP \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}
\caption{The additional fixed linear reference, fitted with source-only preprocessing and stochastic gradient descent. Entries are percentage mean $\pm$ sample SD across the five corresponding source partitions. Repeated ranking metrics share the same cohort and scores.}
\label{tab:linear}
\end{table*}''')
    # The complete seed-level linear table and optimization receipts remain compact.
    rows=[]
    per=pd.read_csv(ROOT/'results/linear_per_seed.csv')
    for s in SEEDS:
        r=read(ROOT/f'logs/linear_{s}_complete.json');a=row(per,seed=s,group='primary',source_fpr_target=.01);b=row(per,seed=s,group='primary',source_fpr_target=.001)
        rows.append(f"{s} & {r['iterations']} & {r['transformed_dimensions']} & {100*a.TPR:.4f} & {100*a.FPR:.4f} & {100*b.TPR:.4f} & {100*b.FPR:.4f}"+r' \\')
    write('linear_seed_supplement.tex',r'''\begin{center}\small
\begin{tabular}{rrrrrrr}
\toprule
Seed & Epochs & Dimensions & TPR (1\%) & FPR (1\%) & TPR (0.1\%) & FPR (0.1\%) \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}\end{center}''')
    threshold_rows=[]
    for s in SEEDS:
        receipt=read(ROOT/f'logs/linear_{s}_complete.json')
        for alpha in [.01,.001]:
            threshold_rows.append(f"{s} & {100*alpha:g} & {receipt['thresholds'][str(alpha)]:.12f} & {100*receipt['validation_fpr'][str(alpha)]:.5f}"+r' \\')
    write('linear_threshold_supplement.tex',r'''\begin{center}\small
\begin{tabular}{llrr}
\toprule
Linear seed & Source target (\%) & Logit threshold & Validation FPR (\%) \\
\midrule
'''+ '\n'.join(threshold_rows)+r'''
\bottomrule\end{tabular}\end{center}''')
    original_per=pd.read_csv(ROOT.parent/'results/per_seed_results.csv').query('cohort=="external_primary"').sort_values(['variant','seed','source_fpr_target'])
    pr=[]
    for _,r in original_per.iterrows():
        view='P' if r['variant']=='prefix2480' else 'F'
        pr.append(f"{view} & {int(r.seed)} & {100*r.source_fpr_target:g} & {100*r.ROC_AUC:.3f} & {100*r.AP:.3f} & {100*r.TPR:.3f} & {100*r.FPR:.3f}"+r' \\')
    write('primary_seed_tables.tex',r'''\begin{center}\small
\begin{tabular}{lrrrrrr}
\toprule
View & Seed & Source target (\%) & AUC & AP & TPR & FPR \\
\midrule
'''+ '\n'.join(pr)+r'''
\bottomrule\end{tabular}\end{center}''')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.8),layout='constrained')
    groups=['primary','DLL_0','DLL_1','age_0-30','age_31-365','age_over365'];labels=['Primary','Non-DLL','DLL','Age 0-30 d','Age 31-365 d','Age >365 d']
    sg=pd.read_csv(ROOT.parent/'results/subgroup_summary.csv')
    # Use original, unchanged subgroup summaries for the tree reference.
    for j,alpha in enumerate([.01,.001]):
        for learner,color,offset,marker in [('LightGBM','#2a6085',-.13,'o'),('Logistic Regression','#9b5125',.13,'s')]:
            vals=[];errs=[]
            for group in groups:
                if learner=='Logistic Regression':r=row(linear,group=group,source_fpr_target=alpha)
                elif group=='primary':r=row(cut,positive_minimum=5,source_fpr_target=alpha)
                elif group.startswith('age_'):r=row(age,positive_minimum=5,report_age=group[4:],source_fpr_target=alpha)
                else:
                    # Original subgroup keys are explicitly resolved, not inferred by score.
                    r=row(sg,variant='prefix2480',grouping='is_dll',group=group[-1],source_fpr_target=alpha)
                vals.append(100*r.FPR_mean);errs.append(100*r.FPR_sd)
            axes[j].errorbar(vals,np.arange(len(groups))+offset,xerr=errs,fmt=marker,color=color,label=learner,capsize=2,markersize=4)
        axes[j].set_yticks(np.arange(len(groups)),labels);axes[j].invert_yaxis();axes[j].set_xlabel('External FPR (%)');axes[j].set_title(f'Source-validation target: {100*alpha:g}%');axes[j].axvline(100*alpha,color='.45',ls=':',lw=1);axes[j].grid(axis='x',alpha=.15)
    axes[1].legend(loc='lower right',fontsize=8,frameon=False)
    fig.savefig(FIG/'learner_subgroup_fpr.pdf');fig.savefig(FIG/'learner_subgroup_fpr.png',dpi=220);plt.close(fig)
    save(ROOT/'logs/paper_revision_assets.json',{'status':'COMPLETE','generated_files':{p.name:sha(p) for p in G.glob('*main.tex')},'figure_sha256':sha(FIG/'learner_subgroup_fpr.pdf')})
if __name__=='__main__':build()
