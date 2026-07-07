"""Aggregate the multi-seed mixed-domain kill-test into mean+/-std tables with seed-level
paired significance. Reads FW_s{0-4}_{dec,cpl,mix}.json, LL_s{0-4}_{dec,cpl,mix}.json (synthetic)
and real_s{0-4}_{cpl,mix}.json (real LOLv2). Run in the server working dir."""
import json, numpy as np
from scipy import stats
SEEDS = [0, 1, 2, 3, 4]

import os as _os
RESULTS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'latex', 'results')  # repo-anchored; override via env RESULTS_DIR

def load(tag):
    import os
    rdir = os.environ.get('RESULTS_DIR', RESULTS_DIR)
    for path in [os.path.join(rdir, tag + '.json'), tag + '.json']:
        try: return json.load(open(path))
        except FileNotFoundError: pass
    return None

def ms(a, p=3):
    a = np.asarray(a, float)
    if len(a) == 0: return 'NA'
    if len(a) == 1: return f'{a[0]:.{p}f}(n1)'
    return f'{a.mean():.{p}f}+/-{a.std(ddof=1):.{p}f}'

# ---- synthetic tracks ----
for track, label in [('FW', 'COMPOSITE (low_rain+low_haze_rain)'), ('LL', 'LL specialist (low_noise, pg+dark)')]:
    C = {m: [] for m in ['dec', 'cpl', 'mix']}; D = {m: [] for m in ['dec', 'cpl', 'mix']}
    for m in ['dec', 'cpl', 'mix']:
        for s in SEEDS:
            d = load(f'{track}_s{s}_{m}')
            if d: C[m].append(d['coupled_test_PSNR']); D[m].append(d['decoupled_test_PSNR'])
    print(f'\n=== {label}  PSNR mean+/-std ===')
    print(f'{"mode":6}{"coupled-test":17}{"decoupled-test":17}{"worst-case":17}{"cross-avg":15} n')
    R = {}
    for m in ['dec', 'cpl', 'mix']:
        c = np.array(C[m]); d = np.array(D[m]); n = min(len(c), len(d))
        if n == 0: continue
        c, d = c[:n], d[:n]; w = np.minimum(c, d); a = (c + d) / 2; R[m] = (c, d, w, a)
        print(f'{m:6}{ms(c):17}{ms(d):17}{ms(w):17}{ms(a):15} {n}')
    if {'dec', 'cpl', 'mix'} <= set(R):
        n = min(len(R['mix'][2]), len(R['dec'][2]), len(R['cpl'][2]))
        mixw = R['mix'][2][:n]; specw = np.maximum(R['dec'][2][:n], R['cpl'][2][:n])
        g = mixw - specw; p = stats.ttest_rel(mixw, specw)[1] if n > 1 else float('nan')
        print(f'  -> mix worst-case beats best-specialist worst-case by {g.mean():+.3f} dB (paired p={p:.3g}, n={n})')
        taxc = (R['mix'][0][:n] - R['cpl'][0][:n]).mean(); taxd = (R['mix'][1][:n] - R['dec'][1][:n]).mean()
        print(f'  -> mix in-domain tax: coupled-home {taxc:+.3f}, decoupled-home {taxd:+.3f} dB')

# ---- real LOLv2 ----
print('\n=== REAL LOLv2 (n=100 imgs/seed; mean+/-std over seeds) ===')
RP = {m: [] for m in ['dec', 'cpl', 'mix']}; RS = {m: [] for m in ['dec', 'cpl', 'mix']}
for s in SEEDS:
    rc = load(f'real_s{s}_cpl'); rm = load(f'real_s{s}_mix')
    if rc:
        RP['dec'].append(rc['decoupled']['PSNR']); RS['dec'].append(rc['decoupled']['SSIM'])
        RP['cpl'].append(rc['coupled']['PSNR']); RS['cpl'].append(rc['coupled']['SSIM'])
    if rm:
        RP['mix'].append(rm['coupled']['PSNR']); RS['mix'].append(rm['coupled']['SSIM'])
for m in ['dec', 'cpl', 'mix']:
    print(f'{m:6} PSNR {ms(RP[m]):16} SSIM {ms(RS[m], 4)}  n={len(RP[m])}')
dP = np.array(RP['dec']); dS = np.array(RS['dec'])
for m in ['cpl', 'mix']:
    P = np.array(RP[m]); S = np.array(RS[m]); n = min(len(P), len(dP))
    if n > 1:
        pp = stats.ttest_rel(P[:n], dP[:n])[1]; ps = stats.ttest_rel(S[:n], dS[:n])[1]
        print(f'  -> {m}-dec: dPSNR {(P[:n]-dP[:n]).mean():+.3f} (seed-paired p={pp:.3g})  '
              f'dSSIM {(S[:n]-dS[:n]).mean():+.4f} (p={ps:.3g})  n={n}')
print()
