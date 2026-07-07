"""stats_ci.py - seed-paired 95% CIs + exact sign-flip permutation p-values for the
headline comparisons. n=5, so the sign-flip permutation null has exactly 2^5=32 points
and the permutation p-value is exact (no sampling). Complements the seed-paired t-tests
by giving effect-size intervals and a non-parametric check.

Run: python code/stats_ci.py   (auto-locates latex/results/; override with RESULTS_DIR)
"""
import os, json
import numpy as np
from scipy import stats
from itertools import product

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('RESULTS_DIR', os.path.join(_HERE, '..', 'latex', 'results'))


def load(t):
    return json.load(open(os.path.join(RES, t + '.json')))


def report(diffs, name):
    d = np.asarray(diffs, float); n = len(d); m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    lo, hi = m - tcrit * se, m + tcrit * se
    tp = stats.ttest_rel(d, np.zeros(n))[1]              # paired t vs 0 == one-sample t on diffs
    obs = abs(m)
    cnt = sum(1 for s in product([1, -1], repeat=n) if abs((d * np.array(s)).mean()) >= obs - 1e-12)
    pperm = cnt / (2 ** n)
    print(f'{name:38s} mean={m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  '
          f't-p={tp:.3g}  perm-p={pperm:.3g}')
    return dict(mean=m, lo=lo, hi=hi, t_p=tp, perm_p=pperm)


print('=== Seed-paired effect sizes: 95% CI (t-based) + exact sign-flip permutation p ===\n')

# 1) Composite: mixed worst-case minus best-specialist worst-case (main cure result)
FW = {m: {'cpl': [], 'dec': []} for m in ['dec', 'cpl', 'mix']}
for s in range(5):
    for m in ['dec', 'cpl', 'mix']:
        d = load(f'FW_s{s}_{m}')
        FW[m]['cpl'].append(d['coupled_test_PSNR']); FW[m]['dec'].append(d['decoupled_test_PSNR'])
mix_wc = [min(FW['mix']['cpl'][s], FW['mix']['dec'][s]) for s in range(5)]
best_sp = [max(min(FW['dec']['cpl'][s], FW['dec']['dec'][s]),
               min(FW['cpl']['cpl'][s], FW['cpl']['dec'][s])) for s in range(5)]
report([mix_wc[s] - best_sp[s] for s in range(5)], 'Composite mixed_wc - best_specialist_wc')

# 2) Real LOLv2 (variance-matched)
LP = {'dec': [], 'cpl': [], 'mix': []}
for s in range(5):
    rc = load(f'realv_s{s}_cpl'); rm = load(f'realv_s{s}_mix')
    LP['dec'].append(rc['decoupled']['PSNR']); LP['cpl'].append(rc['coupled']['PSNR'])
    LP['mix'].append(rm['coupled']['PSNR'])
report([LP['cpl'][s] - LP['dec'][s] for s in range(5)], 'LOLv2 coupled - decoupled (PSNR)')
report([LP['mix'][s] - LP['dec'][s] for s in range(5)], 'LOLv2 mixed - decoupled (PSNR)')

# 3) Architecture: CoupleNet - agnostic NAFNet (coupled test)
ag = [load(f'F2ag_s{s}')['coupled_test_PSNR'] for s in range(5)]
cn = [load(f'F2cn_s{s}')['coupled_test_PSNR'] for s in range(5)]
report([cn[s] - ag[s] for s in range(5)], 'Architecture CoupleNet - NAFNet (PSNR)')

# 4) ExDark detection: coupled - decoupled AP
report([load(f'exd_cpl_s{s}')['ap_exdark'] - load(f'exd_dec_s{s}')['ap_exdark'] for s in range(5)],
       'ExDark coupled - decoupled (AP)')

# 5) Deployment (Table f3deploy): real-only vs coupled+real mixed fine-tuning (ExDark AP)
report([load(f'p3_real_s{s}')['ap_exdark'] - load(f'p3_mixed_s{s}')['ap_exdark'] for s in range(5)],
       'Deployment real - mixed (ExDark AP)')
