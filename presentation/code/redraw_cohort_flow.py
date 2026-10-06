"""Redraw the cohort diagram from the unchanged qualification counts only."""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
audit_path = ROOT / 'qualification/input_audit.json'
summary_path = ROOT / 'descriptive_inputs/summary.json'
audit = json.loads(audit_path.read_text())
summary = json.loads(summary_path.read_text())
steps = [('Inventory', summary['quality']['rows'])] + [
    (row['condition'], row['after']) for row in audit['sequential_exclusions']]
names = ['Retrieved\ninventory', 'Binary\nlabels', 'Three\nPE groups',
         'First submission\navailable', 'SHA-disjoint\nprimary cohort']
assert len(steps) == len(names) == 5
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
                     'font.size': 9, 'text.color': 'black',
                     'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
fig = plt.figure(figsize=(7.1, 2.9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 14.2), ylim=(0, 5.8))
ax.axis('off')
width, gap, start = 2.28, 0.65, 0.10
for i, ((_, count), name) in enumerate(zip(steps, names)):
    x = start + i * (width + gap)
    ax.add_patch(Rectangle((x, 3.78), width, 1.45,
                           facecolor='white', edgecolor='black', linewidth=0.75))
    ax.text(x + width / 2, 4.505, name + f'\n{count:,}',
            ha='center', va='center', fontsize=8.3, linespacing=1.2)
    if i < 4:
        ax.add_patch(FancyArrowPatch((x + width + 0.09, 4.505),
                                    (x + width + gap - 0.09, 4.505),
                                    arrowstyle='->', mutation_scale=8,
                                    shrinkA=0, shrinkB=0,
                                    linewidth=0.75, color='black'))
    if i:
        excluded = audit['sequential_exclusions'][i - 1]['excluded']
        ax.text(x + width / 2, 3.43, f'Excluded: {excluded:,}',
                ha='center', va='center', fontsize=8.0)

ax.text(7.1, 2.64,
        'Fixed subsets of the primary cohort; the original inventory is retained',
        ha='center', va='center', fontsize=9)
subsets = [('Temporal condition', 'temporal_sensitivity'),
           ('Identical-TLSH control', 'identical_tlsh_sensitivity'),
           ('Both conditions', 'temporal_and_tlsh_sensitivity')]
for x, (name, key) in zip([0.60, 5.18, 9.76], subsets):
    ax.add_patch(Rectangle((x, 0.56), 3.84, 1.35,
                           facecolor='white', edgecolor='black', linewidth=0.75))
    ax.text(x + 1.92, 1.235, f'{name}\n{audit["cohorts"][key]["rows"]:,} records',
            ha='center', va='center', fontsize=9, linespacing=1.25)

out = ROOT / 'paper/figures'
fig.savefig(out / 'cohort_flow.pdf')
fig.savefig(out / 'cohort_flow.png', dpi=300)
plt.close(fig)
receipt = {'operation': 'Presentation-only redraw; no analyses rerun',
           'style': 'Black and white; serif type; rectangular outlines; no fills or color accents',
           'box_gap_axis_units': gap, 'box_width_axis_units': width,
           'steps': [{'name': name.replace('\n', ' '), 'rows': count}
                     for name, (_, count) in zip(names, steps)],
           'subsets': {key: audit['cohorts'][key]['rows'] for _, key in subsets},
           'input_sha256': {}}
for p in [audit_path, summary_path]:
    with p.open('rb') as f:
        receipt['input_sha256'][p.relative_to(ROOT).as_posix()] = hashlib.file_digest(f, 'sha256').hexdigest()
logs = ROOT / 'presentation/verification'
logs.mkdir(parents=True, exist_ok=True)
(logs / 'figure_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
