"""
gen_pairs.py - Figure 1: matched decoupled/coupled CCDM sample pairs with x4 difference maps.

Same scenes, synthesis seeds and file names as part 1 of gen_figs.py. --pgnoise renders the noise
with the Poisson-Gaussian model stated in the paper (variance a*x+b), under which the coupled noise
is strongest in bright regions, as the Figure 1 caption describes. For each scene the script prints
the correlation between the coupled noise sigma map and the local signal (positive under
Poisson-Gaussian noise, negative under the legacy sigma0/L branch) so the caption can be checked.

Run (server, composite-restore dir):  python3 gen_pairs.py --pgnoise --outdir figs_pg
"""
import os, glob, argparse
import numpy as np, cv2
import ccdm

TYPES = ('low', 'haze', 'rain', 'noise')
IDXS = [3, 7, 12, 19, 27, 41]


def main(a):
    os.makedirs(a.outdir, exist_ok=True)
    params = {'noise_model': 'poisson'} if a.pgnoise else None
    cleans = sorted(glob.glob('clean_test100/*.png'))
    rains = sorted(glob.glob('OneRestore/syn_data/data/rain_mask/*'))
    snows = sorted(glob.glob('OneRestore/syn_data/data/snow_mask/*'))
    print('noise model:', 'Poisson-Gaussian' if a.pgnoise else 'legacy sigma0/L')
    for k, i in enumerate(IDXS):
        cf = cleans[i]
        J = cv2.imread(cf).astype(np.float32) / 255.
        d = np.load('depth/' + os.path.basename(cf).replace('.png', '.npy'))
        H, W = J.shape[:2]; H -= H % 8; W -= W % 8; J = J[:H, :W]; d = d[:H, :W]
        rm = cv2.imread(rains[i % len(rains)]).astype(np.float32) / 255.
        sm = cv2.imread(snows[i % len(snows)]).astype(np.float32) / 255.
        o, g = {}, {}
        for mode in ('decoupled', 'coupled'):
            rng = np.random.default_rng(1000 + i)          # SAME seed both modes -> matched marginals
            lq, gt = ccdm.degrade(J, d, rng, mode=mode, types=TYPES, rain_mask=rm, snow_mask=sm,
                                  params=params)
            o[mode] = np.clip(lq, 0, 1); g[mode] = gt
        diff = np.clip(np.abs(o['coupled'] - o['decoupled']) * 4.0, 0, 1)   # x4 amplified
        cv2.imwrite(os.path.join(a.outdir, f's{k}_clean.png'), (J * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(a.outdir, f's{k}_dec.png'), (o['decoupled'] * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(a.outdir, f's{k}_cpl.png'), (o['coupled'] * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(a.outdir, f's{k}_diff.png'), (diff * 255).astype(np.uint8))
        sig = g['coupled']['sigma'].ravel(); lum = cv2.blur(o['coupled'].mean(2), (15, 15)).ravel()
        r = float(np.corrcoef(sig, lum)[0, 1]) if sig.std() > 0 else float('nan')
        print(f'scene {k} {os.path.basename(cf)} {J.shape}  mean|cpl-dec|={np.abs(o["coupled"] - o["decoupled"]).mean():.4f}'
              f'  mean sigma dec/cpl={g["decoupled"]["sigma"].mean():.4f}/{g["coupled"]["sigma"].mean():.4f}'
              f'  corr(sigma_cpl, local intensity)={r:+.2f}')


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--pgnoise', action='store_true')
    P.add_argument('--outdir', default='figs_pg')
    main(P.parse_args())
