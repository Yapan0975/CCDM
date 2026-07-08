"""Aggregate the multi-seed ExDark real-detection runs (experiment A) into mean+/-std with
seed-level paired significance vs the decoupled baseline. Reads exd_{dec,cpl,mix}_s{0-4}.json."""
import json, numpy as np
from scipy import stats
SEEDS = [0, 1, 2, 3, 4]
M = {'dec': 'decoupled', 'cpl': 'coupled', 'mix': 'mixed', 'cleanft': 'clean-COCO-ft'}

import os as _os
RESULTS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'latex', 'results')  # repo-anchored; override via env RESULTS_DIR

def load(t):
    import os
    rdir = os.environ.get('RESULTS_DIR', RESULTS_DIR)
    for path in [os.path.join(rdir, t + '.json'), t + '.json']:
        try: return json.load(open(path))
        except FileNotFoundError: pass
    return None

def ms(a, p=4):
    a = np.asarray(a, float)
    if len(a) == 0: return 'NA'
    if len(a) == 1: return f'{a[0]:.{p}f}(n1)'
    return f'{a.mean():.{p}f}+/-{a.std(ddof=1):.{p}f}'

data = {k: {'ap': [], 'ap50': [], 'ar': [], 'syn': []} for k in M}
for ab in M:
    for s in SEEDS:
        d = load(f'exd_{ab}_s{s}')
        if d:
            data[ab]['ap'].append(d['ap_exdark']); data[ab]['ap50'].append(d['ap50_exdark'])
            data[ab]['ar'].append(d['ar_exdark']); data[ab]['syn'].append(d['ar_coupled_syn'])

print('=== ExDark REAL low-light detection (mean+/-std over seeds) ===')
print(f'{"mode":10}{"AP@[.5:.95]":18}{"AP50":18}{"AR(cls-agnostic)":20}{"syn-AR":16} n')
ots = load('exd_offtheshelf')
if ots:
    print(f'{"off-shelf":10}{ots["ap_exdark"]:<18.4f}{ots["ap50_exdark"]:<18.4f}'
          f'{ots["ar_exdark"]:<20.4f}{ots["ar_coupled_syn"]:<16.4f} 1(ref)')
for ab in ['cleanft', 'dec', 'cpl', 'mix']:
    print(f'{M[ab]:10}{ms(data[ab]["ap"]):18}{ms(data[ab]["ap50"]):18}{ms(data[ab]["ar"]):20}{ms(data[ab]["syn"]):16} {len(data[ab]["ap"])}')

print('\n--- seed-paired tests vs decoupled ---')
for metric, lab in [('ap', 'AP'), ('ap50', 'AP50'), ('ar', 'AR')]:
    dec = np.array(data['dec'][metric])
    for ab in ['cpl', 'mix', 'cleanft']:
        v = np.array(data[ab][metric]); n = min(len(v), len(dec))
        if n > 1:
            p = stats.ttest_rel(v[:n], dec[:n])[1]
            print(f'  {M[ab]:9}- decoupled  {lab:5}: {(v[:n]-dec[:n]).mean():+.4f}  (seed-paired p={p:.3g}, n={n})')
print()
