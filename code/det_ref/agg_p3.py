"""Aggregate the closing-the-gap deployment runs (Table 10) into mean+/-std with a seed-level
paired test of coupled-synthesis+real (mixed) vs real-only. Reads p3_{real,mixed}_s{0-4}.json.

Each JSON holds ap_exdark / ap50_exdark / ar_exdark for one seed of one arm:
  real   = detector fine-tuned on the real ExDark train split (5890 imgs) only,
  mixed  = 4000 coupled-synthetic + 5890 real ExDark.
The off-the-shelf and synthesis-only references (0.293 / 0.106) are printed from agg_exdark.py."""
import json, os, numpy as np
from scipy import stats

SEEDS = [0, 1, 2, 3, 4]
ARMS = {'real': 'real ExDark only', 'mixed': 'coupled synth + real'}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'latex', 'results')  # repo-anchored; override via env RESULTS_DIR


def load(tag):
    rdir = os.environ.get('RESULTS_DIR', RESULTS_DIR)
    for path in [os.path.join(rdir, tag + '.json'), tag + '.json']:
        try:
            return json.load(open(path))
        except FileNotFoundError:
            pass
    return None


def ms(a, p=4):
    a = np.asarray(a, float)
    if len(a) == 0:
        return 'NA'
    if len(a) == 1:
        return f'{a[0]:.{p}f}(n1)'
    return f'{a.mean():.{p}f}+/-{a.std(ddof=1):.{p}f}'


data = {k: {'ap': [], 'ap50': [], 'ar': []} for k in ARMS}
for arm in ARMS:
    for s in SEEDS:
        d = load(f'p3_{arm}_s{s}')
        if d:
            data[arm]['ap'].append(d['ap_exdark'])
            data[arm]['ap50'].append(d['ap50_exdark'])
            data[arm]['ar'].append(d['ar_exdark'])

print('=== Closing the synthetic-to-real gap on ExDark (mean+/-std over seeds) === [Table 10]')
print(f'{"arm":24}{"AP@[.5:.95]":20}{"AP50":20}{"AR(cls-agnostic)":22} n')
for arm in ['real', 'mixed']:
    print(f'{ARMS[arm]:24}{ms(data[arm]["ap"]):20}{ms(data[arm]["ap50"]):20}'
          f'{ms(data[arm]["ar"]):22} {len(data[arm]["ap"])}')

print('\n--- seed-paired test: mixed (coupled synth + real) - real-only ---')
for metric, lab in [('ap', 'AP'), ('ap50', 'AP50'), ('ar', 'AR')]:
    real = np.array(data['real'][metric])
    mix = np.array(data['mixed'][metric])
    n = min(len(real), len(mix))
    if n > 1:
        diff = mix[:n] - real[:n]
        p = stats.ttest_rel(mix[:n], real[:n])[1]
        n_lower = int((diff < 0).sum())
        print(f'  {lab:5}: mixed-real = {diff.mean():+.4f}  '
              f'(raw seed-paired p={p:.3g}, n={n}; mixed lower on {n_lower}/{n} seeds)')
print('\nNote: raw p-values only. Bonferroni threshold across the seven paper tests is 0.007;')
print('the exact sign-flip permutation floor at n=5 is p=2/32=0.0625, so no n=5 test can fall below it.')
