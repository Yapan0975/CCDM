"""
verify_lambda.py - correctness check for the continuous coupling strength lam.

Three assertions, all on real CCD test scenes:
  A. lam=0.0 reproduces mode='decoupled' pixel-for-pixel (same rng seed)
  B. lam=1.0 reproduces mode='coupled'   pixel-for-pixel (same rng seed)
  C. marginal severity is held fixed across lam (mean noise sigma), so a lam sweep varies
     coupling STRUCTURE only, not degradation strength.
Also reports how the rendered image moves monotonically from the decoupled to the coupled
endpoint as lam increases (mean |I_lam - I_dec| should rise, |I_lam - I_cpl| should fall).

Run: python3 verify_lambda.py            (legacy sigma0/L reference noise branch)
     python3 verify_lambda.py --pgnoise  (Poisson-Gaussian noise, the model used in the paper)
"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import ccdm
from dataset_ccd import COMBOS

CLEAN = 'clean_test100'; DEPTH = 'depth'
RAIN = './OneRestore/syn_data/data/rain_mask'; SNOW = './OneRestore/syn_data/data/snow_mask'


def render(J, d, rm, sm, types, idx, *, mode=None, lam=None, params=None):
    rng = np.random.default_rng(10000 + idx)
    kw = dict(types=types, rain_mask=rm, snow_mask=sm, params=params)
    if lam is not None:
        return ccdm.degrade(J, d, rng, lam=lam, **kw)
    return ccdm.degrade(J, d, rng, mode=mode, **kw)


def main(a):
    params = {'noise_model': 'poisson'} if a.pgnoise else None
    print('noise model:', 'Poisson-Gaussian' if a.pgnoise else 'legacy sigma0/L')
    cleans = sorted(glob.glob(os.path.join(CLEAN, '*.png')))[:a.n]
    rains = sorted(glob.glob(os.path.join(RAIN, '*')))
    snows = sorted(glob.glob(os.path.join(SNOW, '*')))
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    max_err_dec = max_err_cpl = 0.0
    dist_dec = {l: [] for l in lams}; dist_cpl = {l: [] for l in lams}
    sev = {l: {'sigma': []} for l in lams}

    for idx, cf in enumerate(cleans):
        J = cv2.imread(cf).astype(np.float32) / 255.0
        H, W = J.shape[:2]; H -= H % 8; W -= W % 8; J = J[:H, :W]
        d = np.load(os.path.join(DEPTH, os.path.basename(cf).replace('.png', '.npy')))[:H, :W]
        rm = cv2.imread(rains[idx % len(rains)]).astype(np.float32) / 255.0
        sm = cv2.imread(snows[idx % len(snows)]).astype(np.float32) / 255.0
        for cname, types in COMBOS.items():
            if cname == 'low_noise':
                continue
            I_dec, _ = render(J, d, rm, sm, types, idx, mode='decoupled', params=params)
            I_cpl, _ = render(J, d, rm, sm, types, idx, mode='coupled', params=params)
            for l in lams:
                I_l, gt = render(J, d, rm, sm, types, idx, lam=l, params=params)
                if l == 0.0:
                    max_err_dec = max(max_err_dec, float(np.abs(I_l - I_dec).max()))
                if l == 1.0:
                    max_err_cpl = max(max_err_cpl, float(np.abs(I_l - I_cpl).max()))
                dist_dec[l].append(float(np.abs(I_l - I_dec).mean()))
                dist_cpl[l].append(float(np.abs(I_l - I_cpl).mean()))
                sev[l]['sigma'].append(float(gt['sigma'].mean()))

    print(f'=== A/B. endpoint reproduction (max abs pixel error over {len(cleans)} scenes x 2 combos) ===')
    print(f'  lam=0.0 vs mode=decoupled : {max_err_dec:.3e}   {"PASS" if max_err_dec < 1e-6 else "FAIL"}')
    print(f'  lam=1.0 vs mode=coupled   : {max_err_cpl:.3e}   {"PASS" if max_err_cpl < 1e-6 else "FAIL"}')
    print('\n=== C. marginal severity across lam (mean noise sigma; should be ~constant) ===')
    base = np.mean(sev[0.0]['sigma'])
    for l in lams:
        v = np.mean(sev[l]['sigma'])
        print(f'  lam={l:<5g} mean sigma = {v:.6f}  (rel. dev from lam=0: {100*(v-base)/base:+.3f}%)')
    print('\n=== monotone traverse from decoupled to coupled endpoint ===')
    print('  lam    mean|I_lam-I_dec|   mean|I_lam-I_cpl|')
    for l in lams:
        print(f'  {l:<5g}  {np.mean(dist_dec[l]):.6f}          {np.mean(dist_cpl[l]):.6f}')
    ok = (max_err_dec < 1e-6 and max_err_cpl < 1e-6
          and all(abs(np.mean(sev[l]['sigma']) - base) / base < 0.02 for l in lams))
    print('\nOVERALL:', 'PASS' if ok else 'CHECK FAILED')


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--pgnoise', action='store_true')
    P.add_argument('--n', type=int, default=8)
    main(P.parse_args())
