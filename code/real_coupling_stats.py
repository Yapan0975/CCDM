"""
real_coupling_stats.py - measure the illumination-noise coupling of REAL low-light images
and place them on the coupling axis (review item: a measurable real statistic, not an AP
ordering, that shows which synthesis structure real data is closer to).

Method (per scene, LOLv2-real Input/GT pairs):
  1. tile the luminance image into non-overlapping patches;
  2. keep patches that are texture-free in the WELL-EXPOSED reference (GT patch std below
     the --hom_q quantile of the scene) and unclipped in the low image -- in such patches
     the low-image variance is dominated by sensor noise, not scene texture;
  3. regress patch variance on patch mean (Theil-Sen, robust; OLS as a check) and record
     the slope, plus Spearman rank correlation between mean and variance.
Under the coupled (Poisson-Gaussian, signal-dependent) model the slope is positive; under
the decoupled homoscedastic convention it is ~0. The SIGN of the dependence is invariant
to the sRGB encoding (a monotone map), so measuring in released sRGB is legitimate.

With --synthetic, the same statistic is computed on CCDM low_noise renders of the CCD test
scenes (coupled and decoupled, LL-track params: dark gamma 3-5, Poisson-Gaussian), using
the clean image as the reference -- the synthetic ends of the axis.

Run (server, composite-restore dir; CPU is fine):
  python3 -u real_coupling_stats.py --synthetic --out real_coupling_stats.json
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from scipy import stats

LUMA = np.array([0.114, 0.587, 0.299], np.float32)      # BGR weights


def patch_stats(low, ref, patch, hom_q):
    """Return (mu, var) arrays over reference-homogeneous, unclipped patches.

    The variance is computed after removing a per-patch best-fit plane, so smooth
    illumination gradients inside a patch (which scale with brightness and would fake a
    signal-dependence) do not contribute; and patches touching the clip limits are
    excluded, so zero-censoring of the noise (which suppresses variance in the darkest
    patches) cannot induce a spurious positive slope either. Both guards apply equally
    to the real pairs and the synthetic renders.
    """
    H, W = low.shape
    H -= H % patch; W -= W % patch
    lo = low[:H, :W].reshape(H // patch, patch, W // patch, patch).transpose(0, 2, 1, 3)
    rf = ref[:H, :W].reshape(H // patch, patch, W // patch, patch).transpose(0, 2, 1, 3)
    lo = lo.reshape(-1, patch * patch); rf = rf.reshape(-1, patch * patch)
    ref_std = rf.std(1)
    mu = lo.mean(1)
    # per-patch plane detrend: residual = P - G (G^+ P), G = [1, x, y]
    yy, xx = np.mgrid[0:patch, 0:patch]
    G = np.stack([np.ones(patch * patch), xx.ravel(), yy.ravel()], 1)      # (k^2, 3)
    Gpinv = np.linalg.pinv(G)                                               # (3, k^2)
    resid = lo - (lo @ Gpinv.T) @ G.T
    var = (resid ** 2).sum(1) / (patch * patch - 3)
    hom = ref_std <= np.quantile(ref_std, hom_q)
    ok = hom & (mu > 0.01) & (mu < 0.9) & (lo.max(1) < 0.98) & (lo.min(1) > 1.0 / 255.0)
    return mu[ok], var[ok]


def scene_slope(mu, var, min_patches=30):
    """Theil-Sen slope of var~mu (+ OLS slope, Spearman rho); None if too few patches."""
    if len(mu) < min_patches or np.ptp(mu) < 1e-4:
        return None
    ts = stats.theilslopes(var, mu)
    ols = stats.linregress(mu, var)
    rho = stats.spearmanr(mu, var)
    return dict(n=int(len(mu)), slope_ts=float(ts[0]), slope_ols=float(ols.slope),
                intercept_ts=float(ts[1]), rho=float(rho[0]), rho_p=float(rho[1]))


def summarise(name, recs):
    recs = [r for r in recs if r is not None]
    sl = np.array([r['slope_ts'] for r in recs]); rho = np.array([r['rho'] for r in recs])
    npos = int((sl > 0).sum()); n = len(sl)
    if n:
        try:
            sign_p = float(stats.binomtest(npos, n, 0.5).pvalue)
        except AttributeError:                                  # scipy < 1.7
            sign_p = float(stats.binom_test(npos, n, 0.5))
    else:
        sign_p = float('nan')
    boot = [np.median(np.random.default_rng(b).choice(sl, n)) for b in range(2000)] if n else [np.nan]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out = dict(n_scenes=n, slope_median=float(np.median(sl)), slope_ci=[float(lo), float(hi)],
               frac_positive=round(npos / max(1, n), 3), sign_p=sign_p,
               rho_median=float(np.median(rho)))
    print(f"[{name}] scenes={n}  median Theil-Sen slope={out['slope_median']:.4g} "
          f"CI[{lo:.4g},{hi:.4g}]  frac(slope>0)={out['frac_positive']}  sign-p={sign_p:.3g}  "
          f"median rho={out['rho_median']:.3f}", flush=True)
    return out


def run_real(a):
    lows = sorted(glob.glob(os.path.join(a.root, a.low_dir, '*')))
    recs = []
    for lf in lows:
        lo = cv2.imread(lf).astype(np.float32) / 255.0
        gt = cv2.imread(os.path.join(a.root, a.high_dir, os.path.basename(lf))).astype(np.float32) / 255.0
        mu, var = patch_stats(lo @ LUMA, gt @ LUMA, a.patch, a.hom_q)
        recs.append(scene_slope(mu, var))
    return recs


def run_synthetic(a, mode):
    import ccdm
    from dataset_ccd import COMBOS
    params = dict(gamma=(3.0, 5.0), noise_model='poisson')      # LL-track: --dark --pgnoise
    cleans = sorted(glob.glob(os.path.join(a.clean_test, '*.png')))
    rains = sorted(glob.glob(os.path.join(a.rain, '*')))
    snows = sorted(glob.glob(os.path.join(a.snow, '*')))
    recs = []
    for idx, cf in enumerate(cleans):
        J = cv2.imread(cf).astype(np.float32) / 255.0
        H, W = J.shape[:2]; H -= H % 8; W -= W % 8; J = J[:H, :W]
        d = np.load(os.path.join(a.depth_test, os.path.basename(cf).replace('.png', '.npy')))[:H, :W]
        rng = np.random.default_rng(10000 + idx)                # same convention as build_eval
        rm = cv2.imread(rains[idx % len(rains)]).astype(np.float32) / 255.0
        sm = cv2.imread(snows[idx % len(snows)]).astype(np.float32) / 255.0
        lq, _ = ccdm.degrade(J, d, rng, mode=mode, types=COMBOS['low_noise'],
                             rain_mask=rm, snow_mask=sm, params=params)
        lq = np.clip(lq, 0, 1).astype(np.float32)
        mu, var = patch_stats(lq @ LUMA, J @ LUMA, a.patch, a.hom_q)
        recs.append(scene_slope(mu, var))
    return recs


def main(a):
    out = dict(patch=a.patch, hom_q=a.hom_q)
    real = run_real(a)
    out['real_lolv2'] = dict(summary=summarise('REAL LOLv2', real), scenes=real)
    if a.synthetic:
        for mode in ('coupled', 'decoupled'):
            recs = run_synthetic(a, mode)
            out[f'ccdm_{mode}'] = dict(summary=summarise(f'CCDM {mode}', recs), scenes=recs)
    json.dump(out, open(a.out, 'w'))
    print('saved', a.out, flush=True)


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--root', default='lolv2_test')
    P.add_argument('--low_dir', default='Input')
    P.add_argument('--high_dir', default='GT')
    P.add_argument('--patch', type=int, default=8)
    P.add_argument('--hom_q', type=float, default=0.2)
    P.add_argument('--synthetic', action='store_true')
    P.add_argument('--clean_test', default='clean_test100')
    P.add_argument('--depth_test', default='depth')
    P.add_argument('--rain', default='./OneRestore/syn_data/data/rain_mask')
    P.add_argument('--snow', default='./OneRestore/syn_data/data/snow_mask')
    P.add_argument('--out', default='real_coupling_stats.json')
    main(P.parse_args())
