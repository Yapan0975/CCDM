"""Aggregate P1.1 (variance-matched LL) + P1.3 (F2 multi-seed + params/FLOPs)."""
import json, numpy as np, torch, sys
from scipy import stats
sys.path.insert(0, '.')
SEEDS = [0, 1, 2, 3, 4]

import os as _os
RESULTS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'latex', 'results')  # repo-anchored; override via env RESULTS_DIR

def load(t):
    import os
    rdir = os.environ.get('RESULTS_DIR', RESULTS_DIR)
    for path in [os.path.join(rdir, t + '.json'), t + '.json']:
        try: return json.load(open(path))
        except FileNotFoundError: pass
    return None

def ms(a, p=3):
    a = np.asarray(a, float)
    if len(a) == 0: return 'NA'
    if len(a) == 1: return f'{a[0]:.{p}f}(n1)'
    return f'{a.mean():.{p}f}+/-{a.std(ddof=1):.{p}f}'

# ===== P1.1 variance-matched LL =====
print('=== P1.1 variance-matched LL (low_noise, pg+dark+VARMATCH) PSNR mean+/-std ===')
print(f'{"mode":6}{"coupled-test":17}{"decoupled-test":17}{"worst-case":17} n')
RL = {}
for m in ['dec', 'cpl', 'mix']:
    C = []; D = []
    for s in SEEDS:
        d = load(f'LLv_{m}_s{s}')
        if d: C.append(d['coupled_test_PSNR']); D.append(d['decoupled_test_PSNR'])
    if C:
        c = np.array(C); dd = np.array(D); RL[m] = (c, dd, np.minimum(c, dd))
        print(f'{m:6}{ms(c):17}{ms(dd):17}{ms(np.minimum(c,dd)):17} {len(C)}')
if {'dec', 'cpl', 'mix'} <= set(RL):
    n = min(len(RL['mix'][2]), len(RL['dec'][2]), len(RL['cpl'][2]))
    mixw = RL['mix'][2][:n]; specw = np.maximum(RL['dec'][2][:n], RL['cpl'][2][:n])
    p = stats.ttest_rel(mixw, specw)[1] if n > 1 else float('nan')
    print(f'  -> mix worst-case beats best-specialist by {(mixw-specw).mean():+.3f} dB (p={p:.3g}, n={n})')
print('--- P1.1 real LOLv2 (varmatch) ---')
RP = {m: [] for m in ['dec', 'cpl', 'mix']}; RS = {m: [] for m in ['dec', 'cpl', 'mix']}
for s in SEEDS:
    rc = load(f'realv_s{s}_cpl'); rm = load(f'realv_s{s}_mix')
    if rc:
        RP['dec'].append(rc['decoupled']['PSNR']); RS['dec'].append(rc['decoupled']['SSIM'])
        RP['cpl'].append(rc['coupled']['PSNR']); RS['cpl'].append(rc['coupled']['SSIM'])
    if rm:
        RP['mix'].append(rm['coupled']['PSNR']); RS['mix'].append(rm['coupled']['SSIM'])
for m in ['dec', 'cpl', 'mix']:
    print(f'  {m}: PSNR {ms(RP[m])}  SSIM {ms(RS[m],4)}  n={len(RP[m])}')
dP = np.array(RP['dec'])
for m in ['cpl', 'mix']:
    P = np.array(RP[m]); n = min(len(P), len(dP))
    if n > 1:
        print(f'    {m}-dec dPSNR {(P[:n]-dP[:n]).mean():+.3f} (seed-paired p={stats.ttest_rel(P[:n],dP[:n])[1]:.3g})')

# ===== P1.3 F2 multi-seed + params/FLOPs =====
print('\n=== P1.3 F2 (coupling-aware vs agnostic, full coupled, 5 seeds) coupled-test PSNR ===')
ag = [d['coupled_test_PSNR'] for d in (load(f'F2ag_s{s}') for s in SEEDS) if d]
cn = [d['coupled_test_PSNR'] for d in (load(f'F2cn_s{s}') for s in SEEDS) if d]
print(f'  agnostic NAFNet            : {ms(ag)}  n={len(ag)}')
print(f'  CoupleNet (coupling-aware) : {ms(cn)}  n={len(cn)}')
n = min(len(ag), len(cn))
if n > 1:
    print(f'  -> CoupleNet - agnostic: {(np.array(cn[:n])-np.array(ag[:n])).mean():+.3f} dB '
          f'(seed-paired p={stats.ttest_rel(cn[:n],ag[:n])[1]:.3g}, n={n})')
try:
    from thop import profile
    from nafnet import NAFNet, CoupleNet
    x = torch.randn(1, 3, 256, 256)
    for nm, mdl in [('agnostic NAFNet', NAFNet(width=32)), ('CoupleNet', CoupleNet(width=32))]:
        fl, pa = profile(mdl, inputs=(x,), verbose=False)
        print(f'  {nm:18} params={pa/1e6:.2f}M  FLOPs={fl/1e9:.2f}G @256x256')
except Exception as e:
    print(f'  [params/FLOPs] skipped (non-blocking); run code/params_flops.py for these. ({e})')
print()

# ===== P1.2 OneRestore retrained on CCD (5-seed, 3 modes) =====
print('=== P1.2 OneRestore retrained on CCD specialization (5 seeds) ===')
print(f'{"mode":12}{"coupled-test":22}{"decoupled-test":22}{"worst-case":22} n')
OR = {}
for m, ab in [('decoupled','dec'), ('coupled','cpl'), ('mixed','mix')]:
    C = []; D = []
    for s in SEEDS:
        d = load(f'OR_{ab}_s{s}')
        if d: C.append(d['coupled_test_PSNR']); D.append(d['decoupled_test_PSNR'])
    OR[ab] = (np.array(C), np.array(D))
    wc = np.minimum(np.array(C), np.array(D)) if C else np.array([])
    print(f'{m:12}{ms(C):22}{ms(D):22}{ms(wc):22} {len(C)}')
if len(OR['dec'][0]) > 1 and len(OR['cpl'][0]) > 1 and len(OR['mix'][0]) > 1:
    n = min(len(OR['mix'][2]) if len(OR['mix']) > 2 else len(OR['mix'][0]),
            len(OR['dec'][0]), len(OR['cpl'][0]))
    mix_wc  = np.minimum(OR['mix'][0][:n], OR['mix'][1][:n])
    best_sp = np.maximum(np.minimum(OR['dec'][0][:n], OR['dec'][1][:n]),
                         np.minimum(OR['cpl'][0][:n], OR['cpl'][1][:n]))
    p = stats.ttest_rel(mix_wc, best_sp)[1] if n > 1 else float('nan')
    print(f'  -> mix worst-case beats best-specialist worst-case by '
          f'{(mix_wc - best_sp).mean():+.3f} dB (p={p:.3g}, n={n})')
print()
